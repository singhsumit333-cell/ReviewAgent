"""
sop_chat.py
Layers 4 + 5 of the SOP Assistant Agent.

This is the file a teammate would actually run. It:
1. Builds the vector store ONCE when the program starts (not per-question -
   fixes the inefficiency from sop_answer.py, where every single question
   was re-loading and re-embedding all 57 chunks from scratch).
2. Keeps a running conversation_history list, so follow-up questions like
   "okay, what do I do next?" still make sense to Claude.
3. Loops, taking new questions until the user types 'exit'.
"""

import os
from dotenv import load_dotenv
from anthropic import Anthropic

from sop_retriever import build_vector_store, retrieve_relevant_sops

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a helpful, knowledgeable teammate answering colleagues'
questions about internal SOPs (Standard Operating Procedures).

For each question, you will be given relevant SOP excerpts. Answer the way an
experienced colleague would explain it out loud - in your own words, clearly
and conversationally. Do not just repeat the SOP text verbatim. If the excerpts
don't actually answer the question, say so honestly rather than guessing.

After your explanation, briefly mention which SOP section it relates to, in
case the teammate wants to look up the exact wording themselves.

Since this is an ongoing conversation, use earlier questions and answers for
context when a teammate asks a follow-up (e.g. "what should I do next?").
"""


def rewrite_question_with_history(question, conversation_history):
    """
    Fixes a real bug: retrieve_relevant_sops() only ever sees the CURRENT
    question text - it has no idea what was discussed earlier. So a vague
    follow-up like "what should I do?" gets embedded and searched exactly
    as-is, with no mention of the actual topic (e.g. cash advancing) - and
    can match the WRONG SOP section entirely.

    This function asks Claude to rewrite the follow-up into a standalone
    question first, using the recent conversation as context, BEFORE we
    do retrieval. E.g. "what should I do?" becomes "what should I do if I
    find cash advancing activity?" - now retrieval has something concrete
    to search for.

    If there's no history yet (first question), there's nothing to rewrite
    against, so we just return the question unchanged.
    """
    if not conversation_history:
        return question

    # Only use the last 2 turns (4 messages) as context - enough to resolve
    # "what about that?" style follow-ups, without the rewrite prompt itself
    # growing huge as the conversation gets long.
    recent_history = conversation_history[-4:]
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent_history)

    rewrite_prompt = f"""Conversation so far:
{history_text}

Latest message: "{question}"

Rewrite the latest message as a standalone question that includes whatever
context from the conversation is needed to understand it on its own, with
no other text. Reply with ONLY the rewritten question, nothing else."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": rewrite_prompt}]
    )

    return response.content[0].text.strip()


def build_augmented_message(question, vector_store, conversation_history):
    """
    For THIS turn only, retrieves relevant SOP chunks and glues them onto
    the ORIGINAL question (so Claude still sees exactly what the teammate
    typed). But retrieval itself now uses the REWRITTEN, standalone version
    of the question - so it searches for what the teammate actually means,
    not just the literal (possibly vague) words they typed.

    We do NOT save this augmented version into conversation_history
    afterward (see the main loop below) - if we did, every past turn's SOP
    excerpts would keep getting resent forever, growing the conversation
    huge and expensive for no benefit.
    """
    search_query = rewrite_question_with_history(question, conversation_history)

    matches = retrieve_relevant_sops(vector_store, search_query)
    context_text = "\n\n---\n\n".join(match.page_content for match in matches)

    return f"""Teammate's question: {question}

Relevant SOP excerpts:
{context_text}"""


def run_chat():
    print("Building knowledge base... (this happens once)")
    vector_store = build_vector_store()
    print("Ready. Type your question, or 'exit' to quit.\n")

    conversation_history = []

    while True:
        question = input("You: ")

        if question.strip().lower() == "exit":
            print("Goodbye!")
            break

        augmented_message = build_augmented_message(question, vector_store, conversation_history)

        # We send: all PAST clean turns + this turn's context-augmented message.
        # This is the one place the augmented (bloated) version is used.
        api_messages = conversation_history + [
            {"role": "user", "content": augmented_message}
        ]

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=api_messages
        )

        answer = response.content[0].text
        print(f"\nAssistant: {answer}\n")

        # Now save the CLEAN version (plain question, plain answer) into
        # history - not the augmented one. This keeps future turns lean
        # while Claude still has the topic/context of what was discussed.
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    run_chat()