from prompts import CHUNK_ANALYST_INSTRUCTIONS

CHUNK_ANALYST = {
    "name": "chunk-analyst",
    "description": (
        "Reads ONE retrieved chunk file from /retrieved/ and returns the facts that "
        "answer the user's question, with a quote and citation. The task description "
        "must include the user's question and exactly one file path."
    ),
    "system_prompt": CHUNK_ANALYST_INSTRUCTIONS,
}

SUBAGENTS = [CHUNK_ANALYST]