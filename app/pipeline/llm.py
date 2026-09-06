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
            options={"temperature": settings.llm_temperature},
        )
        return response["message"]["content"].strip()
