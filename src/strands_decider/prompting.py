"""Turning a (state, question) pair into text, and slots into meaning.

The whole trick of the architecture lives here. The classification heads are
*generic*: head k has no fixed meaning. It scores "whatever option was listed at
position k in the prompt". Nothing in the weights knows that slot 0 is `billing`
in one request and `Calm` in the next -- the prompt alone establishes that binding,
which is what makes the model adapt to novel label sets without retraining.

Two consequences shape everything downstream:

1. The prompt must number the options explicitly and unambiguously, so the
   binding is learnable and stable.
2. Training must shuffle option order (see data/collate.py), or the heads quietly
   memorise "slot 0 tends to be the positive class" and genericity is lost.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    Content,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    derive_confidence,
    derive_score_confidence,
)

if TYPE_CHECKING:
    from tokenizers import Tokenizer

# Rendered on both sides of a noul so the two slots read like any other option list.
NOUL_SLOT_LABELS = ("false", "true")
NOUL_DEFAULT_CRITERIA = {
    "false": "the statement does not hold for this state",
    "true": "the statement holds for this state",
}

# The prompt formats. "numbered" is the default: tagged text, options `1.`, `2.`, ..., read
# at `<answer>`. "lettered" is the tokenizer's chat template with thinking off, a user turn
# with State / Question / Options, the options under letter codes A, B, ..., read at the last
# token of the generation prompt. It needs the tokenizer (`chat_frame`).
RENDERS = ("numbered", "lettered")
LETTERED_SYSTEM = (
    "Classify the supplied state. Select the one option that answers the question. "
    "Reply with only the selected option code."
)
LETTERED_INSTRUCTION = "Return only the letter code of the best option."
NUM_OPTION_CODES = 255  # the schema's option limit


def option_code(k: int) -> str:
    """The k-th letter code (0-based): A ... Z, AA ... AZ, BA ..., as spreadsheet columns."""
    code = ""
    k += 1
    while k:
        k, r = divmod(k - 1, 26)
        code = chr(65 + r) + code
    return code


OPTION_CODES = tuple(option_code(k) for k in range(NUM_OPTION_CODES))  # A ... IU


def chat_frame(tok: Any) -> tuple[str, str]:
    """The chat template text before and after the user turn's content, for render "lettered".

    The template writes its own BOS, so a lettered prompt is tokenised without special tokens.
    """
    mark = "\x00USER\x00"
    text = tok.apply_chat_template(
        [{"role": "system", "content": LETTERED_SYSTEM}, {"role": "user", "content": mark}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    head, sep, tail = text.partition(mark)
    if not sep:
        raise ValueError("the chat template dropped the user content")
    return head, tail


def render_content(content: Content | None) -> str:
    """Flatten a state or instruction into text, stably.

    Dicts and lists are emitted as indented JSON rather than str() so that key
    order and unicode are deterministic -- the same state must always tokenise
    identically, otherwise the shared-prefix cache in infer.py would be unsound.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, indent=2, ensure_ascii=False, sort_keys=False)


@dataclass(frozen=True)
class RenderedQuestion:
    """A question flattened to text plus the slot->label map needed to read it back."""

    text: str
    # slot_labels[k] is the option name bound to head k for this request.
    slot_labels: tuple[str, ...]
    # slot_descriptions[k] is that option's rubric text, kept so a score answer can
    # report a legend of what each level meant rather than bare indices.
    slot_descriptions: tuple[str, ...]
    kind: str  # "noul" | "choice" | "score"
    # (start, end) character span of each option's line within `text`. A pointer head
    # scores option k from the hidden state at the LAST token of this span -- the token
    # that has just read the whole option under causal attention. Characters rather than
    # token indices so callers can map them with the tokeniser's offset mapping instead
    # of rebuilding the prompt segment by segment.
    option_spans: tuple[tuple[int, int], ...] = ()

    @property
    def n_slots(self) -> int:
        return len(self.slot_labels)


def _option_block(
    pairs: Sequence[tuple[str, str]], render: str = "numbered"
) -> tuple[str, list[tuple[int, int]]]:
    """Number options from 1 so slot k <-> the line reading `k+1.`; with render "lettered",
    label them `A: name: description`, so slot k <-> code k.

    Also returns each line's (start, end) character span within the block. Descriptions
    are collapsed to a single line so one option is always exactly one line, which is
    what makes the span unambiguous for a pointer readout.
    """
    lines: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, (name, desc) in enumerate(pairs):
        desc = " ".join((desc or "").split())
        if render == "lettered":
            line = f"{OPTION_CODES[i]}: {name}" + (f": {desc}" if desc else "")
        else:
            line = f"{i + 1}. {name}" + (f" \u2014 {desc}" if desc else "")
        lines.append(line)
        spans.append((cursor, cursor + len(line)))
        cursor += len(line) + 1  # the joining newline
    return "\n".join(lines), spans


def render_question(
    question: Question,
    *,
    option_order: Sequence[int] | None = None,
    render: str = "numbered",
    frame: tuple[str, str] | None = None,
) -> RenderedQuestion:
    """Render one question. `option_order` permutes the options (training augmentation).

    The permutation is applied to the *rendered* list, and slot_labels records the
    resulting binding, so callers never have to track the permutation themselves.
    `render="lettered"` needs `frame` (`chat_frame`); its text ends with the frame's tail.
    """
    _check_render(render, frame)
    instructions = render_content(question.instructions)

    if isinstance(question, NoulQuestion):
        crit = {**NOUL_DEFAULT_CRITERIA, **(question.criteria or {})}
        pairs = [(lbl, render_content(crit[lbl])) for lbl in NOUL_SLOT_LABELS]
        kind = "noul"
        header = "Decide whether the statement is true of the state."
    elif isinstance(question, ChoiceQuestion):
        pairs = [(name, render_content(desc)) for name, desc in question.criteria.items()]
        kind = "choice"
        header = "Select exactly one option."
    elif isinstance(question, ScoreQuestion):
        # Level index is the label; the rubric text is the description.
        pairs = [(str(i), desc) for i, desc in enumerate(question.criteria)]
        kind = "score"
        header = "Rate the state against the ordered levels below (lowest first)."
    else:  # pragma: no cover - guarded by pydantic unions
        raise TypeError(f"unknown question type: {type(question)!r}")

    if option_order is not None:
        if sorted(option_order) != list(range(len(pairs))):
            raise ValueError("option_order must be a permutation of the option indices")
        pairs = [pairs[i] for i in option_order]

    if render == "lettered":
        assert frame is not None
        block, block_spans = _option_block(pairs, render)
        prefix = f"Question:\n{instructions}\n\nOptions:\n"
        text = prefix + block + f"\n\n{LETTERED_INSTRUCTION}" + frame[1]
    else:
        block, block_spans = _option_block(pairs)
        prefix = f'<question type="{kind}">\n{header}\n{instructions}\n<options>\n'
        text = prefix + block + "\n</options>\n</question>\n<answer>"
    base = len(prefix)
    return RenderedQuestion(
        text=text,
        slot_labels=tuple(n for n, _ in pairs),
        slot_descriptions=tuple((d or "").strip() for _, d in pairs),
        kind=kind,
        option_spans=tuple((base + s, base + e) for s, e in block_spans),
    )


def _check_render(render: str, frame: tuple[str, str] | None) -> None:
    if render not in RENDERS:
        raise ValueError(f"unknown render {render!r}")
    if render == "lettered" and frame is None:
        raise ValueError('render "lettered" needs the chat template frame (chat_frame)')


def render_state(
    state: Content, render: str = "numbered", frame: tuple[str, str] | None = None
) -> str:
    """The shared prefix. Everything before this point is identical across the
    questions in one request, which is exactly what infer.py caches once. With render
    "lettered" it is the chat template's head and the user turn's State block."""
    _check_render(render, frame)
    if render == "lettered":
        assert frame is not None
        return f"{frame[0]}State:\n{render_content(state)}\n\n"
    return f"<state>\n{render_content(state)}\n</state>\n"


def build_prompt(state: Content, question: Question, **kw: Any) -> tuple[str, RenderedQuestion]:
    rq = render_question(question, **kw)
    return render_state(state, kw.get("render", "numbered"), kw.get("frame")) + rq.text, rq


def read_noul(probs: Sequence[float], rq: RenderedQuestion) -> float:
    """Recover P(true) regardless of which slot `true` landed in."""
    idx = rq.slot_labels.index("true")
    return float(probs[idx])


def read_choice(probs: Sequence[float], rq: RenderedQuestion) -> dict[str, float]:
    return {name: float(p) for name, p in zip(rq.slot_labels, probs, strict=True)}


def score_legend(rq: RenderedQuestion) -> dict[str, str]:
    """{level index -> rubric text} in canonical ascending order.

    Needed because the rendering may have reversed the rubric; the caller should
    always see levels the way the request declared them.
    """
    return {
        level: desc
        for level, desc in sorted(
            zip(rq.slot_labels, rq.slot_descriptions, strict=True),
            key=lambda kv: int(kv[0]),
        )
    }


def read_score(probs: Sequence[float], rq: RenderedQuestion) -> tuple[float, dict[str, float]]:
    """Expected level index, and the distribution keyed by level.

    Because slots may have been permuted, we re-key by the true level index before
    taking the expectation -- otherwise a shuffled rendering would score backwards.
    """
    by_level = {name: float(p) for name, p in zip(rq.slot_labels, probs, strict=True)}
    expected = sum(int(level) * p for level, p in by_level.items())
    ordered = {str(i): by_level[str(i)] for i in range(len(by_level))}
    return float(expected), ordered


def fit_window(
    tok: Tokenizer,
    state_text: str,
    question_texts: list[str],
    max_length: int,
    *,
    add_special_tokens: bool = True,
) -> tuple[list[int], list[list[int]], list[list[tuple[int, int]]]]:
    """Tokenize a state and its questions, refusing a prompt that does not fit a window of
    `max_length` tokens.

    Returns the state's ids, each question's ids, and each question's character offsets.
    `tok` is a `tokenizers.Tokenizer`, such as a Hugging Face fast tokenizer's
    `backend_tokenizer`. This function turns its padding and truncation off, so callers
    must not run it on one tokenizer from two threads at once. `add_special_tokens=False` encodes the state as it is,
    for a render whose text already holds its BOS (render "lettered").
    """
    tok.no_padding()
    tok.no_truncation()
    encodings = tok.encode_batch(question_texts, add_special_tokens=False)
    q = [e.ids for e in encodings]
    offsets = [list(e.offsets) for e in encodings]
    longest = max(len(x) for x in q)
    s = tok.encode(state_text, add_special_tokens=add_special_tokens).ids
    if len(s) + longest > max_length:
        raise ValueError(
            f"prompt of {len(s) + longest} tokens ({len(s)} state + {longest} question) "
            f"exceeds the context window of {max_length} tokens"
        )
    return s, q, offsets


def option_token_index(
    offsets: Sequence[Sequence[int]],
    spans: Sequence[Sequence[int]],
    base: int,
) -> list[int]:
    """Last token index of each option's line, for a pointer readout.

    The last token of the span is the one that has just read the whole option under
    causal attention. Raises when an option has no surviving token: truncation has
    removed it, and scoring it from a neighbour's representation would be silently
    wrong.
    """
    out: list[int] = []
    for s, e in spans:
        a, b = base + s, base + e
        last = -1
        for j, (lo, hi) in enumerate(offsets):
            if hi <= lo:
                continue
            if lo >= a and hi <= b:
                last = j
        if last < 0:
            raise ValueError(
                f"option span ({a},{b}) has no tokens left; the prompt was "
                "truncated through its option list"
            )
        out.append(last)
    return out


def to_answer(
    rq: RenderedQuestion, probs: list[float], *, ordinal_smoothing: float = 0.0
) -> Answer:
    if rq.kind == "noul":
        return NoulAnswer(noul=round(read_noul(probs, rq), 4))

    if rq.kind == "choice":
        by_name = read_choice(probs, rq)
        best = max(by_name, key=lambda k: by_name[k])
        return ChoiceAnswer(
            choice=best,
            probabilities={k: round(v, 4) for k, v in by_name.items()},
            confidence=round(derive_confidence(list(by_name.values())), 4),
        )

    score, ordered = read_score(probs, rq)
    return ScoreAnswer(
        score=round(score, 4),
        # Report the rubric in canonical ascending order, not the order rendered.
        legend=score_legend(rq),
        probabilities={k: round(v, 4) for k, v in ordered.items()},
        # Ordinal confidence, not max-probability: see derive_score_confidence.
        confidence=round(
            derive_score_confidence(list(ordered.values()), ordinal_smoothing=ordinal_smoothing),
            4,
        ),
    )
