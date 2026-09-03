from typing import Optional

import ollama

from app.config import settings

# Small models like Gemma 4B tend to drift into chatty preambles ("Sure, here's
# the cleaned transcript:") or trailing commentary unless the constraints are
# repeated and made explicit. Sandwiching the instruction (start + end),
# spelling out DO NOT rules, and showing a worked example keeps output
# on-format far more reliably than a single plain instruction does.
_SYSTEM_TEMPLATE = """You are a transcript-cleaning tool. Your ONLY task: {instruction}

Rules:
- Output ONLY the cleaned transcript text. Nothing else.
- Do NOT add a preamble, label, or header (no "Here is the cleaned transcript:", no "Sure,", no "Output:").
- Do NOT add explanations, notes, or commentary before or after the transcript.
- Do NOT wrap the output in quotes or markdown.
- Do NOT ask questions.
- Do NOT change the meaning of the transcript.

Example:
Input: um so like i think we should uh go with option two right
Output: I think we should go with option two.

Reminder, your ONLY task: {instruction}
Respond with the cleaned transcript ONLY — no preamble, no commentary, no quotes."""


class GemmaStage:
    """Stage 2: LLM processing of the Stage 1 transcript via Ollama (Gemma)."""

    def __init__(self) -> None:
        self._client = ollama.Client(host=settings.ollama_host)

    def process(self, transcript: str, instruction: Optional[str] = None) -> str:
        system_prompt = _SYSTEM_TEMPLATE.format(
            instruction=instruction or settings.llm_default_instruction
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
