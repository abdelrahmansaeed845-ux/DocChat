import time
from dotenv import load_dotenv

from loaders import load_files
from chunking import split_documents
from vector_store import build_vector_store
from agent import build_agent

load_dotenv()

# 1) Build the knowledge base
docs = load_files(["files/climate_change.pdf", "files/global_warming.pdf"])
chunks = split_documents(docs)
store = build_vector_store(chunks)
print(f"Indexed {len(chunks)} chunks from {len(docs)} pages\n")

# 2) Build the agent
agent = build_agent(store, k=4, max_concurrent_analysts=2)

QUESTIONS = [
    "hi",
    "How do I bake bread?",
    "Compare what the two documents say about the causes of climate change.",
]


def run(question: str):
    print("=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    for msg in result["messages"]:
        if getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                args = str(call["args"])[:120].replace("\n", " ")
                print(f"  → TOOL CALL: {call['name']}  {args}")
        elif msg.type == "tool":
            print(f"  ← RESULT: {str(msg.content)[:120]}".replace("\n", " "))

    print("\nFINAL ANSWER:\n", result["messages"][-1].content, "\n")


for q in QUESTIONS:
    run(q)
    time.sleep(5)