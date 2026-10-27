from loaders import load_files
from chunking import split_documents
from vector_store import build_vector_store
from tools import make_search_tool

docs = load_files(["files/climate_change.pdf", "files/global_warming.pdf"])
store = build_vector_store(split_documents(docs))

tool = make_search_tool(store, k=4)
print("TOOL NAME:", tool.name)
print()

print("Retrieval test:")
for doc, score in store.similarity_search_with_score("causes of global warming", k=3):
    print(f"  score={score:.3f}  {doc.metadata['name']} p.{doc.metadata.get('page_number')}")
    print("   ", doc.page_content[:120].replace("\n", " "))