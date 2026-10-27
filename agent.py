from langchain_groq import ChatGroq
from deepagents import create_deep_agent

from prompts import build_system_prompt
from subagents import SUBAGENTS
from tools import make_search_tool


def build_agent(store, k: int = 5, max_concurrent_analysts: int = 3):
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.1,
        max_tokens=4096,
        max_retries=2,
    )
    return create_deep_agent(
        model=llm,
        tools=[make_search_tool(store, k=k)],
        subagents=SUBAGENTS,
        system_prompt=build_system_prompt(max_concurrent_analysts),
    )

