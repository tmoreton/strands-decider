"""The token readout and the lettered render (modeling.TokenReadoutHead, prompting render
"lettered"), and the pointer path they must leave unchanged.

CPU only, with test_ddp.py's tiny Qwen3 torso and test-trained tokenizer, and a stand-in
chat template. The Qwen3.5 and Gemma 4 checks run where those tokenizers are in the local
Hugging Face cache.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
import torch
from test_ddp import _examples, _tiny_model_factory, _tokenizer_file

import training.train as T
from strands_decider.infer import EngineConfig, SystemOneEngine
from strands_decider.modeling import (
    StrandsDeciderConfig,
    StrandsDeciderModel,
    TokenReadoutHead,
    pool_last_token,
)
from strands_decider.prompting import (
    OPTION_CODES,
    build_prompt,
    chat_frame,
    option_code,
    render_question,
    render_state,
)
from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest
from training.data.collate import CollatorConfig, SystemOneCollator
from training.data.format import load_examples, write_jsonl
from training.soup import soup, soup_heads

REAL_CLIP = torch.nn.utils.clip_grad_norm_
# A stand-in chat template with Gemma 4's layout (system turn, user turn, empty thought).
TEMPLATE = (
    "<bos>{% for m in messages %}<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|turn>model\n<|channel>thought\n<channel|>"
    "{% endif %}"
)
QUESTIONS = {
    "team": ChoiceQuestion(
        instructions="Which team?", criteria={"billing": "money", "tech": "bugs", "sales": "deals"}
    ),
    "urgent": NoulQuestion(instructions="It is urgent."),
    "mood": ScoreQuestion(instructions="How angry?", criteria=["calm", "annoyed", "furious"]),
}
# What `_pointer_trace` gave on origin/main 03bce59, before the token readout existed. The
# renders and batches are exact; the float values hold to the platform's rounding.
MAIN_RENDERS = "bb75baf142839488b12ab5331f4d7b7c804d48d90c3e574cf71a8a9afc39960f"
MAIN_BATCHES = "7d2ad3f1c9b937565d247346e57ba050e9858f26b43e9aedd32a2d06bf94d44d"
MAIN_LOSSES = [1.2146382, 1.0987360, 1.3282480, 1.2259021, 1.1267848, 1.0307044]
MAIN_PROBS = [0.2462, 0.3807, 0.3731, 0.5091, 0.4687, 0.2211, 0.3102] * 2  # cached, batched


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("token_readout_corpus")
    rows = _examples(150, seed=1)
    write_jsonl(str(d / "a.jsonl"), rows)
    write_jsonl(str(d / "val.jsonl"), _examples(20, seed=2))
    _tokenizer_file(str(d / "tokenizer.json"), rows)
    _tokenizer_file(str(d / "other_tokenizer.json"), _examples(60, seed=7), vocab_size=300)
    return d


def _factory(corpus):
    build = _tiny_model_factory(str(corpus / "tokenizer.json"))

    def make(config, **kw):
        model = build(config, **kw)
        model.tokenizer.chat_template = TEMPLATE
        return model

    return make


def _token_readout(corpus):
    return _factory(corpus)(
        StrandsDeciderConfig(base_model="tiny", head_type="token_readout", render="lettered")
    )


def _pointer(corpus, tokenizer="tokenizer.json"):
    cfg = StrandsDeciderConfig(base_model="tiny", pointer_dim=16)
    return _tiny_model_factory(str(corpus / tokenizer))(cfg)


def _batches():
    return [list(range(i, i + 8)) for i in range(0, 48, 8)]


# ---------------------------------------------------------------- the pointer path


def _pointer_trace(corpus):
    """The pointer path end to end, on names origin/main has: (sha256 of the numbered
    renders, sha256 of six training batches, their losses, the engine's probabilities on
    both serving paths)."""
    renders = hashlib.sha256()
    for ex in _examples(150, seed=1):
        for order in (None, list(reversed(range(ex.n_options)))):
            renders.update(build_prompt(ex.state, ex.to_question(), option_order=order)[0].encode())
    rows = load_examples([str(corpus / "a.jsonl")])
    torch.manual_seed(0)
    model = _pointer(corpus)
    coll = SystemOneCollator(model.tokenizer, CollatorConfig(max_length=2048, seed=0), train=True)
    batches, losses = hashlib.sha256(), []
    for idx in _batches():
        b = coll([rows[i] for i in idx])
        for k in sorted(b):
            batches.update(f"{k}{b[k].dtype}{tuple(b[k].shape)}".encode())
            batches.update(b[k].contiguous().numpy().tobytes())
        with torch.no_grad():
            out = model(**{k: b[k] for k in b if k not in ("teacher", "has_teacher")})
        losses.append(float(out["loss"]))
    engine = SystemOneEngine(model, EngineConfig(device="cpu"))
    probs = []
    for prefix_cache in (True, False):
        engine.cfg.use_prefix_cache = prefix_cache
        got = engine.evaluate(SystemOneRequest(state="Payouts failing.", questions=QUESTIONS))
        for a in got.answers.values():
            probs += [a.noul] if a.type == "noul" else list(a.probabilities.values())
    return renders.hexdigest(), batches.hexdigest(), losses, probs


def test_the_pointer_path_is_what_origin_main_computes(corpus):
    renders, batches, losses, probs = _pointer_trace(corpus)
    assert renders == MAIN_RENDERS
    assert batches == MAIN_BATCHES
    assert losses == pytest.approx(MAIN_LOSSES, rel=1e-5)
    assert probs == pytest.approx(MAIN_PROBS, abs=2e-4)  # answers are rounded to 4 places


def test_a_numbered_config_is_saved_without_a_render_key(tmp_path):
    """strands-decider 0.2.0 refuses unknown config keys, so a pointer checkpoint's JSON keeps
    the keys it had; a config without the key loads as render "numbered"."""
    assert "render" not in json.loads(StrandsDeciderConfig().to_json())
    p = tmp_path / "cfg.json"
    p.write_text(StrandsDeciderConfig(head_type="token_readout", render="lettered").to_json())
    assert StrandsDeciderConfig.from_json(str(p)).render == "lettered"
    p.write_text(json.dumps({"base_model": "x", "head_type": "pointer"}))
    assert StrandsDeciderConfig.from_json(str(p)).render == "numbered"


# ---------------------------------------------------------------- render


def test_option_codes():
    assert [option_code(k) for k in (0, 1, 23, 25, 26, 27, 51, 52, 254)] == [
        "A", "B", "X", "Z", "AA", "AB", "AZ", "BA", "IU",
    ]  # fmt: skip
    assert len(set(OPTION_CODES)) == len(OPTION_CODES) == 255


def test_lettered_render_golden():
    frame = ("<H>", "<T>")
    q = ChoiceQuestion(
        instructions="Which team owns this ticket?",
        criteria={"billing": "money  matters\nonly", "tech": "", "sales": "deals"},
    )
    rq = render_question(q, option_order=[2, 0, 1], render="lettered", frame=frame)
    text = render_state({"ticket": "refund"}, "lettered", frame) + rq.text
    assert text == (
        '<H>State:\n{\n  "ticket": "refund"\n}\n\n'
        "Question:\nWhich team owns this ticket?\n\n"
        "Options:\nA: sales: deals\nB: billing: money matters only\nC: tech\n\n"
        "Return only the letter code of the best option.<T>"
    )
    assert rq.slot_labels == ("sales", "billing", "tech")
    assert [rq.text[a:b] for a, b in rq.option_spans] == [
        "A: sales: deals",
        "B: billing: money matters only",
        "C: tech",
    ]
    with pytest.raises(ValueError, match="frame"):
        render_question(q, render="lettered")
    with pytest.raises(ValueError, match="unknown render"):
        render_question(q, render="bulleted")


def test_chat_frame_from_a_chat_template(corpus):
    head, tail = chat_frame(_token_readout(corpus).tokenizer)
    assert head.startswith("<bos><|turn>system\nClassify the supplied state.")
    assert head.endswith("<|turn>user\n")
    assert tail == "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"


@pytest.mark.parametrize("name", ["Qwen/Qwen3.5-2B-Base", "google/gemma-4-E2B-it"])
def test_codes_a_to_x_are_single_tokens_in_the_prompt(name):
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        pytest.skip(f"the {name} tokenizer is not in the local Hugging Face cache")
    frame = chat_frame(tok)
    q = ChoiceQuestion(instructions="Pick one.", criteria={f"o{k}": f"text {k}" for k in range(24)})
    rq = render_question(q, render="lettered", frame=frame)
    text = render_state("a state", "lettered", frame) + rq.text
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    if tok.bos_token_id is not None:  # Gemma's template writes one BOS; Qwen3.5 has none
        assert enc["input_ids"].count(tok.bos_token_id) == 1
    starts = {lo: i for i, (lo, _) in enumerate(enc["offset_mapping"])}
    base = len(text) - len(rq.text)
    for k, (a, _) in enumerate(rq.option_spans):
        assert len(tok.encode(OPTION_CODES[k], add_special_tokens=False)) == 1, OPTION_CODES[k]
        assert tok.decode([enc["input_ids"][starts[base + a]]]) == OPTION_CODES[k]
    # the shared-prefix cache tokenises state and question apart: the same ids as whole
    s = tok(render_state("a state", "lettered", frame), add_special_tokens=False)["input_ids"]
    assert s + tok(rq.text, add_special_tokens=False)["input_ids"] == enc["input_ids"]


# ---------------------------------------------------------------- readout


def test_token_readout_head_is_255_rows_masked_and_trimmed(corpus):
    model = _token_readout(corpus)
    assert isinstance(model.head, TokenReadoutHead)
    assert {k: tuple(v.shape) for k, v in model.head.state_dict().items()} == {
        "proj.weight": (255, 32)
    }
    rows = load_examples([str(corpus / "a.jsonl")])[:8]
    cfg = CollatorConfig(render="lettered")
    b = SystemOneCollator(model.tokenizer, cfg, train=False)(rows)
    bos = model.tokenizer("<bos>", add_special_tokens=False)["input_ids"][0]
    assert (b["input_ids"][:, 0] == bos).all()  # the template's BOS, no second one
    out = model(**{k: b[k] for k in ("input_ids", "attention_mask", "n_slots", "opt_idx")})
    width = int(b["n_slots"].max())
    assert out["logits"].shape == (8, width)
    hidden = model.encode(b["input_ids"], b["attention_mask"])
    want = pool_last_token(hidden, b["attention_mask"]).float() @ model.head.proj.weight.t()
    assert torch.allclose(out["logits"], want[:, :width], atol=1e-6)
    for r, n in enumerate(b["n_slots"].tolist()):
        assert torch.isneginf(out["log_probs"][r, n:]).all()
        assert out["log_probs"][r, :n].exp().sum().item() == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(ValueError, match="255 slots"):
        model.head(torch.zeros(1, 32), torch.zeros(1, 256, 32))


def test_lettered_prompt_past_max_length_is_refused(corpus):
    model = _token_readout(corpus)
    rows = load_examples([str(corpus / "a.jsonl")])[:8]
    coll = SystemOneCollator(
        model.tokenizer, CollatorConfig(max_length=40, render="lettered"), train=False
    )
    with pytest.raises(ValueError, match="longer than max_length"):
        coll(rows)


def test_token_readout_soup_is_the_mean_of_the_logits(corpus, tmp_path):
    members = []
    for s in range(3):
        torch.manual_seed(s)
        m = _token_readout(corpus)
        m.save_pretrained(str(tmp_path / f"s{s}"))
        members.append(m.head)
    soup([str(tmp_path / f"s{s}") for s in range(3)], str(tmp_path / "soup"))
    cfg = json.load(open(tmp_path / "soup" / "strands_decider_config.json", encoding="utf-8"))
    assert (cfg["head_type"], cfg["render"], cfg["pointer_dim"]) == (
        "token_readout",
        "lettered",
        256,
    )
    head = TokenReadoutHead(32)
    from strands_decider.modeling import load_head_state

    head.load_state_dict(load_head_state(str(tmp_path / "soup")))
    x, o = torch.randn(5, 32), torch.zeros(5, 7, 32)
    want = torch.stack([m(x, o) for m in members]).mean(0)
    assert torch.allclose(head(x, o), want, atol=1e-6)
    assert set(soup_heads([m.state_dict() for m in members], "token_readout")) == {"proj.weight"}


# ---------------------------------------------------------------- serving


def test_token_readout_engine_answers_as_the_model_does_on_both_paths(corpus):
    model = _token_readout(corpus).eval()
    engine = SystemOneEngine(model, EngineConfig(device="cpu"))
    frame = chat_frame(model.tokenizer)
    want = {}
    for name, q in QUESTIONS.items():
        prompt, rq = build_prompt("some state text", q, render="lettered", frame=frame)
        enc = model.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        base = len(prompt) - len(rq.text)
        opt = SystemOneCollator.option_token_index(enc["offset_mapping"], rq.option_spans, base)
        with torch.no_grad():
            lp = model(
                input_ids=torch.tensor([enc["input_ids"]]),
                attention_mask=torch.ones(1, len(enc["input_ids"]), dtype=torch.long),
                n_slots=torch.tensor([rq.n_slots]),
                opt_idx=torch.tensor([opt]),
            )["log_probs"][0]
        want[name] = dict(zip(rq.slot_labels, lp.exp().tolist(), strict=True))
    for prefix_cache in (True, False):
        engine.cfg.use_prefix_cache = prefix_cache
        got = engine.evaluate(SystemOneRequest(state="some state text", questions=QUESTIONS))
        for name, a in got.answers.items():
            if a.type == "noul":
                assert a.noul == pytest.approx(want[name]["true"], abs=1e-4), name
            else:
                assert a.probabilities == pytest.approx(want[name], abs=1e-4), name


# ---------------------------------------------------------------- training


def test_reference_renders_its_own_way_with_the_student_option_order(corpus):
    """A token readout student with a pointer reference: the reference reads its own numbered
    render, with the student's option order, and its own tokenizer."""
    torch.manual_seed(0)
    student, ref_model = _token_readout(corpus), _pointer(corpus, "other_tokenizer.json").eval()
    rows = load_examples([str(corpus / "a.jsonl")])
    for i, ex in enumerate(rows):  # a teacher distribution makes each slot identifiable
        ex.teacher = [(j + 1) / 100 + i for j in range(ex.n_options)]
    stu_cfg = CollatorConfig(max_length=2048, render="lettered", seed=0)
    ref = T._frozen_reference(student, rows, _batches(), stu_cfg, "cpu", ref_model)
    stu_coll = SystemOneCollator(student.tokenizer, stu_cfg, train=True)  # training's renders
    ref_coll = SystemOneCollator(
        ref_model.tokenizer, CollatorConfig(max_length=2048, seed=0), train=True
    )
    for m, idx in enumerate(_batches()):
        batch = [rows[i] for i in idx]
        s, b = stu_coll(batch), ref_coll(batch)
        for k in ("labels", "n_slots", "teacher", "label_dist"):
            assert torch.equal(s[k], b[k]), (m, k)  # the same permutation of every row
        stu_text = student.tokenizer.batch_decode(s["input_ids"], skip_special_tokens=True)
        ref_text = ref_model.tokenizer.batch_decode(b["input_ids"], skip_special_tokens=True)
        for r, ex in enumerate(batch):  # letter codes for the student, numbers for the reference
            label, name = int(s["labels"][r]), ex.options[ex.label][0]
            assert f"\n{OPTION_CODES[label]}: {name}" in stu_text[r]
            assert f"\n{label + 1}. {name}" in ref_text[r]
        with torch.no_grad():
            lp = ref_model(
                input_ids=b["input_ids"],
                attention_mask=b["attention_mask"],
                n_slots=b["n_slots"],
                opt_idx=b["opt_idx"],
                temperature=1.0,
            )["log_probs"]
        for r, n in enumerate(b["n_slots"].tolist()):
            assert torch.equal(ref[m, r, :n], lp[r, :n])
            assert torch.isneginf(ref[m, r, n:]).all()


def test_token_readout_student_trains_adapter_and_readout_against_a_pointer_reference(
    corpus, tmp_path, monkeypatch
):
    torch.manual_seed(1)
    ref_model = _pointer(corpus, "other_tokenizer.json").eval()
    factory = _factory(corpus)
    monkeypatch.setattr(StrandsDeciderModel, "from_pretrained_base", staticmethod(factory))
    monkeypatch.setattr(StrandsDeciderModel, "load", staticmethod(lambda path, **k: ref_model))
    grads = []

    def clip(params, max_norm, *a, **k):
        params = list(params)
        grads.append([p.grad.detach().clone() for p in params])
        return REAL_CLIP(params, max_norm, *a, **k)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    out = tmp_path / "run"
    cfg = T.TrainConfig(
        train_files=[str(corpus / "a.jsonl")],
        val_files=[str(corpus / "val.jsonl")],
        base_model="tiny",
        head_type="token_readout",
        render="lettered",
        kl_frozen_weight=0.3,
        kl_frozen_reference="ref",
        precompute_frozen_kl=True,
        head_dropout=0.0,
        max_length=2048,
        micro_batch_size=8,
        grad_accum=4,
        max_steps=2,
        group_by_length=True,
        length_group_mega=2,
        log_every=1,
        eval_every=0,
        output_dir=str(out),
    )
    T.train(cfg)
    history = json.load(open(out / "history.json", encoding="utf-8"))
    assert all(h["kl"] > 0 for h in history if "kl" in h)
    saved = StrandsDeciderConfig.from_json(str(out / "strands_decider_config.json"))
    assert (saved.head_type, saved.render) == ("token_readout", "lettered")
    # gradient reaches the readout (only the rows of codes in use) and the adapter
    named = [n for n, p in _token_readout(corpus).named_parameters() if p.requires_grad]
    for step in grads:
        by = dict(zip(named, step, strict=True))
        head = by["head.proj.weight"]
        assert head[:12].abs().sum() > 0 and head[12:].abs().sum() == 0  # corpus max 12 options
        assert sum(float(g.abs().sum()) for n, g in by.items() if "lora_" in n) > 0


def test_from_yaml_takes_a_token_readout_config(tmp_path):
    import yaml

    p = tmp_path / "token_readout.yaml"
    p.write_text(
        yaml.safe_dump({"head_type": "token_readout", "render": "lettered", "num_slots": 24})
    )
    cfg = T.TrainConfig.from_yaml(str(p))
    assert (cfg.head_type, cfg.render) == ("token_readout", "lettered")


# ---------------------------------------------------------------- export


def test_a_token_readout_export_loads_as_token_readout(corpus, tmp_path, monkeypatch):
    from test_hf_export import _export

    from training import hf_export

    torch.manual_seed(0)
    model = _token_readout(corpus).eval()
    model.config.base_model = hf_export.BASE_MODEL  # the export names a known base
    ckpt = tmp_path / "ckpt"
    model.save_pretrained(str(ckpt))
    (ckpt / "train_config.json").write_text("{}")
    out = tmp_path / "out"
    assert _export(tmp_path, ckpt, out) == "written"
    hf_export.verify(str(out))
    torso = copy.deepcopy(model.torso).unload()  # the base torso, without the adapter
    monkeypatch.setattr(StrandsDeciderModel, "_load_torso", staticmethod(lambda *a: torso))
    back = StrandsDeciderModel.load(str(out))
    assert (back.config.head_type, back.config.render) == ("token_readout", "lettered")
    assert torch.equal(back.head.proj.weight, model.head.proj.weight)
    request = SystemOneRequest(state="some state text", questions=QUESTIONS)
    a, b = (SystemOneEngine(m, EngineConfig(device="cpu")) for m in (model, back))
    assert b.frame == chat_frame(model.tokenizer)
    assert b.evaluate(request).answers == a.evaluate(request).answers


def test_engines_without_token_readout_support_refuse_it(corpus, tmp_path):
    from strands_decider.hidden_engine import save_assets
    from strands_decider.vision import VisionEngine, VisionEngineConfig

    model = _token_readout(corpus)
    with pytest.raises(ValueError, match="numbered"):
        VisionEngine(model, VisionEngineConfig(device="cpu"), image_processor=object())
    with pytest.raises(ValueError, match="pointer head"):
        save_assets(model, str(tmp_path / "assets"), "m")
