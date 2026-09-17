from dataclasses import dataclass


@dataclass(frozen=True)
class Mode:
    description: str
    instruction: str
    format_rules: str
    example_input: str
    example_output: str
    is_diff: bool = False


MODES: dict[str, Mode] = {
    "clean": Mode(
        description="Clean up the transcript: fix punctuation/grammar, remove filler words, keep the meaning unchanged.",
        instruction=(
            "Clean up this raw speech transcript: fix punctuation and grammar, "
            "remove filler words, and keep the meaning unchanged."
        ),
        format_rules=(
            "- Output ONLY the cleaned transcript, as plain prose.\n"
            "- Do NOT wrap the output in quotes or markdown.\n"
            "- Do NOT shorten or summarize — keep every point, just cleaned up.\n"
            "- Do NOT omit or substitute any specific numbers, names, or version "
            "identifiers (e.g. \"5 hours\", \"Gemma 4\") — copy them exactly as heard."
        ),
        example_input="um so like i think we should uh go with option two right",
        example_output="I think we should go with option two.",
    ),
    "clean_fast": Mode(
        description=(
            "Faster alternative to 'clean': Gemma outputs only the find/replace "
            "edits needed (filler removal, grammar/capitalization fixes) instead "
            "of regenerating the whole transcript; edits are applied in Python."
        ),
        instruction=(
            "Find the filler words, stutters, false starts, and grammar/"
            "capitalization mistakes in this raw speech transcript, and output "
            "the minimal find/replace edits needed to fix them. Do not rewrite "
            "or rephrase anything that is already correct."
        ),
        format_rules=(
            "- Output ONLY a JSON array of edit objects, nothing else — no "
            "markdown code fences, no commentary before or after.\n"
            '- Each object has exactly two string fields: "find" and "replace".\n'
            '- "find" MUST be an exact, verbatim substring copied character-for-'
            "character from the transcript — never a paraphrase.\n"
            "- Only include edits for filler words, stutters, false starts, or "
            "clear grammar/capitalization mistakes.\n"
            "- Do NOT include an edit for any part of the transcript that is "
            "already correct.\n"
            "- Do NOT omit or substitute any specific numbers, names, or version "
            "identifiers — those are never something to edit.\n"
            '- Do NOT ever add, remove, or change a negation word ("not", "n\'t", '
            '"never", "no", "none") — if an edit\'s \"find\" span contains a '
            'negation word, that same negation word must appear unchanged, '
            "word-for-word, in \"replace\" too.\n"
            "- If no edits are needed, output exactly: []"
        ),
        example_input="um so like i think we should uh go with option two right",
        example_output=(
            '[{"find": "um so like i think", "replace": "I think"}, '
            '{"find": " uh go", "replace": " go"}, '
            '{"find": "two right", "replace": "two."}]'
        ),
        is_diff=True,
    ),
    "summary": Mode(
        description="Concise summary (2-3 sentences) of what was discussed.",
        instruction="Write a concise summary of what was discussed in this transcript, in 2-3 sentences.",
        format_rules=(
            "- Output ONLY the summary, as plain prose, 2-3 sentences.\n"
            "- Do NOT list action items, decisions, or quotes separately.\n"
            "- Do NOT restate the transcript verbatim or quote it directly."
        ),
        example_input=(
            "we looked at the q3 numbers, revenue is up 12 percent, and we agreed "
            "to hire two more engineers next quarter"
        ),
        example_output=(
            "The team reviewed Q3 numbers, noting a 12% revenue increase, and "
            "agreed to hire two more engineers next quarter."
        ),
    ),
    "action_items": Mode(
        description="Bullet list of action items (tasks someone needs to do).",
        instruction="Extract every action item (a task someone needs to do) mentioned in this transcript.",
        format_rules=(
            '- Output ONLY a bullet list, one action item per line, each line starting with "- ".\n'
            "- Do NOT invent action items that are not clearly stated in the transcript.\n"
            "- Do NOT include decisions, summaries, or anything that is not an action item.\n"
            "- If there are no action items, output exactly: None."
        ),
        example_input="john said he will send the report by friday and we need someone to book the venue",
        example_output="- John will send the report by Friday.\n- Book the venue.",
    ),
    "decisions": Mode(
        description="Bullet list of decisions that were made (not just discussed).",
        instruction="Extract every decision that was made (not merely discussed or proposed) in this transcript.",
        format_rules=(
            '- Output ONLY a bullet list, one decision per line, each line starting with "- ".\n'
            "- Do NOT include action items, open questions, or options that were discussed but not decided.\n"
            "- If no decisions were made, output exactly: None."
        ),
        example_input="we debated option a and b for a while and finally agreed to go with option b",
        example_output="- Went with option B.",
    ),
    "key_quotes": Mode(
        description="Notable or important quotes, copied verbatim.",
        instruction="Extract the most notable or important verbatim quotes from this transcript.",
        format_rules=(
            '- Output ONLY a bullet list, one quote per line, each line starting with "- ", '
            "wrapped in double quotes, copied EXACTLY as spoken.\n"
            "- Do NOT paraphrase or correct grammar inside the quotes.\n"
            "- Do NOT add attribution, commentary, or explanation.\n"
            "- If there are no notable quotes, output exactly: None."
        ),
        example_input="i really think this is the best decision we have made all year",
        example_output='- "i really think this is the best decision we have made all year"',
    ),
}

DEFAULT_MODE = "clean"
