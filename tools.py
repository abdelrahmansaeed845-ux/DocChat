from datetime import datetime, timezone
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command


def make_citation(metadata: dict) -> str:
    """Human-readable citation: 'file.pdf, page 12' (no page for TXT/MD)."""
    name = metadata.get("name", "unknown file")
    page = metadata.get("page_number")
    return f"{name}, page {page}" if page else name


def make_search_tool(store, k: int = 5):
    """Create a search tool bound to one vector store."""

    @tool
    def search_documents(
        query: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Search the user's uploaded documents for passages relevant to a query.

        Saves each matching chunk as a file under /retrieved/ and returns the
        file paths with their citations. Use a specific, self-contained query.
        You can call this tool several times with different queries.
        """
        results = store.similarity_search_with_score(query, k=k)

        if not results:
            return Command(update={"messages": [
                ToolMessage("No relevant passages found.", tool_call_id=tool_call_id)
            ]})

        now = datetime.now(timezone.utc).isoformat()
        files = {}
        summary_lines = []

        for doc, score in results:
            m = doc.metadata
            citation = make_citation(m)
            path = f"/retrieved/chunk_{m['chunk_id']}.md"

            body = (
                f"Citation: {citation}\n"
                f"Relevance score: {score:.2f}\n\n"
                f"{doc.page_content}"
            )

            files[path] = {
                "content": body.split("\n"),
                "created_at": now,
                "modified_at": now,
            }
            summary_lines.append(f"{path}  →  cite as ({citation})")

        return Command(update={
            "files": files,
            "messages": [ToolMessage(
                "Saved passages. Analyze each file, and cite the document name "
                "and page, never the file path:\n" + "\n".join(summary_lines),
                tool_call_id=tool_call_id,
            )],
        })

    return search_documents