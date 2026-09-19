"""
sop_loader.py
Layer 1 of the SOP Assistant Agent.

Reads every .docx and .pdf file inside the sop_docs folder, extracts the
raw text from each, then splits that text into smaller overlapping chunks.

Why chunk at all? An embedding model turns text into numbers representing
meaning - but if you feed it an entire 5-page SOP as one chunk, the meaning
gets "averaged out" and blurry (imagine trying to summarize a whole document
in a single sentence - you lose the specific detail a teammate's question
might be about). Smaller chunks (a paragraph or two) keep each piece
focused on one specific idea, so semantic search can match a question to
the *exact* relevant part of the document, not just "somewhere in this SOP."

"Overlapping" chunks (see chunk_overlap below) means each chunk shares a
little text with its neighbor, so we don't accidentally cut a sentence or
instruction in half at a chunk boundary and lose its meaning.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

SOP_FOLDER = "sop_docs"


def load_documents():
    """
    Loops through every file in SOP_FOLDER, picks the right loader based on
    file extension, and extracts its text. Returns a list of LangChain
    Document objects (each has .page_content = the text, and .metadata =
    info like which file it came from).
    """
    documents = []

    for filename in os.listdir(SOP_FOLDER):
        filepath = os.path.join(SOP_FOLDER, filename)

        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.lower().endswith(".docx"):
            loader = Docx2txtLoader(filepath)
        else:
            print(f"Skipping unsupported file: {filename}")
            continue

        loaded = loader.load()
        documents.extend(loaded)
        print(f"Loaded {filename} ({len(loaded)} page/section(s))")

    return documents


def split_into_chunks(documents):
    """
    Takes the full documents and splits them into smaller overlapping chunks.

    chunk_size=500        -> roughly 500 characters per chunk (a paragraph or two)
    chunk_overlap=50      -> each chunk repeats the last 50 characters of the
                              previous one, so context isn't lost at the edges
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    docs = load_documents()
    print(f"\nTotal documents loaded: {len(docs)}")

    chunks = split_into_chunks(docs)
    print(f"Total chunks after splitting: {len(chunks)}\n")

    # Print the first 2 chunks so you can see what a chunk actually looks like
    for i, chunk in enumerate(chunks[:2]):
        print(f"--- Chunk {i+1} ---")
        print(chunk.page_content)
        print(f"(source: {chunk.metadata})\n")