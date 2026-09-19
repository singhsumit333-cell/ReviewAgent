"""
sop_answer.py
Layer 3 of the SOP Assistant Agent.

Takes a teammate's question, retrieves the most relevant SOP chunks
(using sop_retriever.py), then sends both to Claude with a prompt that
asks it to explain the answer in plain, conversational language - like a
knowledgeable colleague would - instead of just repeating SOP text.

This is the same pattern as llm_reasoner.py in your triage agent (raw
Anthropic API call, build a prompt, get a response) - the only real
difference is *what the prompt asks Claude to do* with the information.
"""

import os
from dotenv import load_dotenv
from anthropic import Anthropic

from sop_retriever import build_vector_store, retrieve_relevant_sops

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def build_prompt(question, matches):
    """
    Combines the retrieved SOP chunks into one block of text, then wraps
    it with instructions telling Claude HOW to use that text - this is
    the exact part that solves your original ask: "don't just read out
    the SOP, explain it like a person would."
    """
    context_text = "\n\n---\n\n".join(match.page_content for match in matches)

    prompt = f"""You are a helpful, knowledgeable teammate answering a colleague's
question about internal SOPs (Standard Operating Procedures).

A teammate asked:
"{question}"

Here are the most relevant SOP excerpts found for this question:

{context_text}

Answer the teammate's question the way an experienced colleague would explain it
out loud - in your own words, clearly and conversationally. Do not just repeat
the SOP text verbatim. If the excerpts don't actually answer the question, say so
honestly rather than guessing.

After your explanation, on a new line, briefly mention which SOP section this
relates to, in case they want to look up the exact wording themselves.
"""
    return prompt


def get_conversational_answer(question):
    """
    Full pipeline for one question:
    1. Build the vector store (embeds all SOP chunks)
    2. Retrieve the most relevant chunks for this specific question
    3. Build a prompt combining the question + those chunks + instructions
    4. Send it to Claude and return its response
    """
    vector_store = build_vector_store()
    matches = retrieve_relevant_sops(vector_store, question)

    prompt = build_prompt(question, matches)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.content[0].text


if __name__ == "__main__":
    question = "I noticed a customer sending lots of small payments to the same person, is that something I should worry about?"

    answer = get_conversational_answer(question)

    print(f"Question: {question}\n")
    print(f"Answer:\n{answer}")