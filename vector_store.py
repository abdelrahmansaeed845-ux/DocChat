from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document


def build_vector_store(chunks: list[Document],) -> InMemoryVectorStore:
    """Create an in-memory vector store from document chunks."""

    if not chunks:
        raise ValueError("No chunks provided.")
    embedding_model = FastEmbedEmbeddings()
    store = InMemoryVectorStore(embedding=embedding_model)
    store.add_documents(chunks)

    return store