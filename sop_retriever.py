"""
sop_retriever.py
Layer 2 of the SOP Assistant Agent.

Takes the chunks from sop_loader.py, converts each one into a vector
(a list of numbers representing its meaning), and stores all of them in
a FAISS vector store - a specialized structure built for quickly finding
which stored vectors are "closest in meaning" to a new question's vector.

This is the exact same embedding + FAISS pattern from rag_retriever.py in
your triage agent project. The only thing that's different is what's being
embedded: real SOP text instead of 4 hand-written policy strings.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from sop_loader import load_documents, split_into_chunks

# This loads a small, free, local AI model whose only job is turning text
# into vectors. "Local" means it runs on your own machine - no API call,
# no cost per use, unlike the Claude/Anthropic calls in llm_reasoner.py.
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vector_store():
    """
    Runs the full pipeline: load PDFs -> split into chunks -> embed each
    chunk -> store all of them in a FAISS vector store.

    FAISS.from_documents() does two things in one call:
    1. Sends every chunk's text through embedding_model to get its vector
    2. Stores each (chunk text + its vector) together, ready for searching
    """
    documents = load_documents()
    chunks = split_into_chunks(documents)

    vector_store = FAISS.from_documents(chunks, embedding_model)
    print(f"Vector store built with {len(chunks)} chunks embedded.")

    return vector_store


def retrieve_relevant_sops(vector_store, question, k=3):
    """
    Given a teammate's question, converts it into a vector (using the same
    embedding_model, so it's comparable to the stored chunks), then asks
    FAISS to return the 'k' chunks whose vectors are closest in meaning.

    k=3 means "give me the 3 most relevant chunks" - not too few (might
    miss the answer if it's spread across chunks) and not too many (would
    dilute the prompt with irrelevant text later).
    """
    results = vector_store.similarity_search(question, k=k)
    return results


if __name__ == "__main__":
    store = build_vector_store()

    # Test question - not using exact SOP wording on purpose, to prove
    # semantic search works even when words don't match exactly
    test_question = "what happens if someone is turning cash-like payments into regular purchases"

    matches = retrieve_relevant_sops(store, test_question)

    print(f"\nQuestion: {test_question}\n")
    for i, match in enumerate(matches):
        print(f"--- Match {i+1} (page {match.metadata.get('page_label')}) ---")
        print(match.page_content)
        print()