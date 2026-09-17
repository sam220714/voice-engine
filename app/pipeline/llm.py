import json
import re
from typing import Optional

import ollama

from app.config import settings
from app.pipeline.modes import DEFAULT_MODE, MODES

# Small models like Gemma 4B tend to drift into chatty preambles ("Sure, here's
# the summary:") or trailing commentary unless the constraints are repeated
# and made explicit. Sandwiching the instruction (start + end), spelling out
# DO NOT rules, and showing a worked example keeps output on-format far more
# reliably than a single plain instruction does.
_SYSTEM_TEMPLATE = """You are a transcript-processing tool. Your ONLY task: {instruction}

Rules:
{format_rules}
- Do NOT add a preamble, label, or header (no "Here is the result:", no "Sure,", no "Output:").
- Do NOT add explanations, notes, or commentary before or after the result.
- Do NOT ask questions.

Example:
Input: {example_input}
Output: {example_output}

Reminder, your ONLY task: {instruction}
Respond with the result ONLY — no preamble, no commentary, no quotes around the whole thing."""


class GemmaStage:
    """Stage 2: LLM processing of the Stage 1 transcript via Ollama (Gemma)."""

    def __init__(self) -> None:
        self._client = ollama.Client(host=settings.ollama_host)

    def process(
        self,
        transcript: str,
        mode: str = DEFAULT_MODE,
        instruction: Optional[str] = None,
    ) -> str:
        mode_cfg = MODES.get(mode, MODES[DEFAULT_MODE])
        system_prompt = _SYSTEM_TEMPLATE.format(
            instruction=instruction or mode_cfg.instruction,
            format_rules=mode_cfg.format_rules,
            example_input=mode_cfg.example_input,
            example_output=mode_cfg.example_output,
        )
        response = self._client.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
            format="json" if mode_cfg.is_diff else None,
            options={"temperature": settings.llm_temperature},
        )
        raw = response["message"]["content"].strip()

        if mode_cfg.is_diff:
            return _apply_edits(transcript, raw)
        return raw


_NEGATION_PATTERN = re.compile(r"\b(?:not|never|no|none)\b|n't", re.IGNORECASE)


def _negation_count(text: str) -> int:
    return len(_NEGATION_PATTERN.findall(text))


def _apply_edits(original: str, edits_json: str) -> str:
    """Apply a JSON list of {"find", "replace"} edits to the original text.

    Two independent safety nets, since a wrong edit here can silently corrupt
    meaning rather than just fail loudly:
    - Any edit whose "find" doesn't appear verbatim in the (still-being-edited)
      text is skipped — the model occasionally paraphrases an anchor instead
      of quoting it exactly.
    - Any edit whose "replace" has a different count of negation words
      ("not", "n't", "never", "no", "none") than its "find" is skipped —
      the prompt forbids this, but a dropped/added negation flips the
      sentence's meaning, so it's also enforced here rather than trusted.
    """
    try:
        edits = json.loads(edits_json)
    except json.JSONDecodeError:
        return original

    text = original
    for edit in edits:
        find = edit.get("find", "")
        replace = edit.get("replace", "")
        if not find or find not in text:
            continue
        if _negation_count(find) != _negation_count(replace):
            continue
        text = text.replace(find, replace, 1)
    return text
