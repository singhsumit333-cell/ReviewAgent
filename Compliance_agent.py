"""
compliance_agent.py
The upgraded, tool-calling version of the assistant.

Instead of our code ALWAYS running SOP retrieval before every question
(what sop_chat.py did), we now give Claude a menu of tools and let IT
decide, per message, whether it needs to search the SOPs, search the web,
both, or neither. This is genuine "agentic" behavior - the AI choosing its
own tools based on the situation, not a fixed pipeline.

Two kinds of tools work differently here:

1. search_sops - a CUSTOM tool (this is OUR function). When Claude wants to
   use it, the API pauses and hands control back to our code. We run the
   real Python function, package up the result, and send it back to
   Claude to continue.

2. web_search - a BUILT-IN "server tool" from Anthropic. When Claude wants
   to search the web, Anthropic's own servers run the search and hand
   Claude the results automatically, all within the same API call. We
   don't write any code to make this happen - it's the one tool in this
   file we DON'T have to manually execute.
"""

import os
from dotenv import load_dotenv
from anthropic import Anthropic

from sop_retriever import build_vector_store, retrieve_relevant_sops

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a compliance assistant for a Financial Crime Operations
team. You help teammates in two ways, and picking the right one matters -
don't default to the heavier one just because a message mentions an account
or a risk pattern.

1. General SOP questions - the teammate wants information, an explanation,
   or a refresher, not a review of a specific account. Answer conversationally,
   in your own words, using the search_sops tool to find relevant policy
   content. Don't just repeat SOP text verbatim - explain it like an
   experienced colleague would. Do NOT ask for case type or case notes for
   these - just answer.

   Examples that are ONLY this, even though they mention accounts/risk terms:
   - "what's the process for a GSR review?"
   - "what counts as cash advancing?"
   - "what are the red flags for ATO?"
   - "what's the difference between hard-linked and soft-linked accounts?"
   These are asking HOW something works in general. Answer directly.

2. Case review help - the teammate has a SPECIFIC account or case in hand
   and wants it assessed. The signal is they're describing something that
   actually happened on an account, not asking how a concept works in
   general - e.g. "I have an account that ...", "can you help me review
   this case", "not sure if this is ATO or cash advancing, here's what I'm
   seeing". If it's ambiguous which of the two this is, ask a single quick
   clarifying question rather than assuming - don't jump straight into a
   full intake questionnaire on a guess.

   Once it's clearly a real case, DO NOT reflexively run a fixed checklist.
   Use whatever the teammate has already told you first:
   - If they've already given specific signals (e.g. "IP changed, then got
     a high-value payment"), use search_sops right away to pull the
     distinguishing criteria for the patterns they're weighing, and tell
     them what those specific signals already indicate per the SOP. Only
     THEN ask for the 1-2 pieces of information still missing to firm up
     the call (e.g. "was there also a profile change - new device, email,
     or payout method?" or "do you have the business/website name?") -
     never re-ask for something they already told you.
   - Only run the full intake (case type, then the full case-notes list:
     account activity, email type, MCC, linked account flags, IP
     anomalies, pattern changes, transaction tier, business name/website -
     already de-identified, no real PII expected) when the teammate has
     given you almost nothing to go on yet and a full assessment is
     clearly what they want.

   The three case types have DIFFERENT goals - don't run the same generic
   flow for all of them. Once you know which type it is, gear the whole
   assessment toward that type's actual objective:

   - GSR (general account review): the goal is judging whether account
     activity itself is suspicious (ATO, cash advancing, structuring, etc).
     Use search_sops for the relevant pattern's criteria, weigh it against
     the account signals given, and only pull in web_search on the business/
     website if the case notes reference one and it's relevant to the call.

   - ZTL (Zettle/POS application review): do the normal GSR-style account
     activity check, PLUS - and this is the actual focus of a ZTL review -
     answer two specific questions using web_search:
       1. Does the business belong to this account holder? Look for a
          public link between the account holder's name and the business
          (ownership/registration records, "about us" info, anything tying
          the person to the business).
       2. Is the business legitimate and does it actually exist? Check:
          - Business registration signals findable via Google
          - Whether the business shows up on Google Maps with a real
            physical location - this matters specifically for ZTL because
            a POS machine implies face-to-face transactions, so a business
            with no discoverable physical location is a real red flag here
            (this check is specific to ZTL, not a general business review)
     Use search_sops for what the SOP requires to approve/deny a ZTL
     application, then end with a clear recommendation: does the evidence
     support that this is a real business genuinely tied to this account
     holder with a real physical presence, or not - that's the actual
     decision a ZTL review exists to make.

   - General business/website check: this is a standalone deep-dive on a
     business/website with no account-activity assessment attached. Always
     cover these specific checks with web_search:
       - Website domain age (how long has the domain existed - a very
         recently registered domain is a flag)
       - Scam/trust rating - search for the domain on ScamAdvisor and
         ScamDetector (or similar scam-checking services) and report what
         rating or flags come up
       - Social media presence - does the business have active, credible
         social accounts, or none/very new ones
       - Reviews and complaints - web_search only returns what your exact
         query text targets, so run several specific searches rather than
         one generic "reviews" search: e.g. "<business/domain> reviews
         trustpilot", "<business/domain> reddit", "<business/domain>
         complaints", "<business/domain> scam" - a single vague query can
         easily miss a platform that a targeted one would catch
     Use search_sops for what the SOP flags as legitimacy red flags for a
     business generally. If the teammate wants something beyond these
     standard checks, ask what else to cover - but don't skip these by
     default, they're the baseline for this case type.

   Once you have enough to work with:
   - Combine SOP guidance and external research into a clear, structured
     assessment appropriate to that case type's actual goal (a suspicion
     call for GSR, an approve/deny recommendation for ZTL, a legitimacy
     verdict for a business check) - not a one-size-fits-all report.
   - Note: web_search only returns search-result snippets - it cannot open
     and read a specific page or social profile directly, so don't imply
     you "checked" a page yourself; say what the search results show and
     what still needs manual verification.
   - Ask if the merchant has provided any supporting documents, since that
     affects the assessment.

Always cite which SOP section your guidance comes from. Be direct and
practical - this is for working investigators, not a general audience.
"""

SOP_TOOL = {
    "name": "search_sops",
    "description": "Search the internal Financial Crime Operations SOPs for guidance on a specific topic, case type, or risk pattern. Use this any time you need policy guidance to answer a question or assess a case.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for, e.g. 'cash advancing investigation steps' or 'account takeover red flags'"
            }
        },
        "required": ["query"]
    }
}

WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 6
}


def execute_sop_search(vector_store, query):
    """
    This is the actual Python function that runs when Claude calls the
    search_sops tool. It just reuses your existing retriever - nothing new
    here, it's the same function from sop_retriever.py.
    """
    matches = retrieve_relevant_sops(vector_store, query)
    return "\n\n---\n\n".join(match.page_content for match in matches)


def call_claude(messages):
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        tools=[SOP_TOOL, WEB_SEARCH_TOOL],
        messages=messages
    )


def get_text_from_response(response):
    """Response content is a list of blocks (text, tool_use, etc.) - this
    pulls out and joins just the readable text blocks."""
    return "".join(block.text for block in response.content if block.type == "text")


def run_chat():
    print("Building knowledge base...")
    vector_store = build_vector_store()
    print("Ready. Type your question, or 'exit' to quit.\n")

    conversation_history = []

    while True:
        question = input("You: ")
        if question.strip().lower() == "exit":
            print("Goodbye!")
            break

        conversation_history.append({"role": "user", "content": question})

        response = call_claude(conversation_history)

        # This loop ONLY handles our custom tool (search_sops). If Claude
        # used web_search instead, that already happened server-side, and
        # stop_reason here will simply be "end_turn" - nothing to loop on.
        while response.stop_reason == "tool_use":
            # Save Claude's turn (including its tool request) into history,
            # exactly as the API sent it - this is required so the next
            # call has full context of what was asked for and why.
            conversation_history.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "search_sops":
                    print(f"  [searching SOPs for: {block.input['query']}]")
                    result_text = execute_sop_search(vector_store, block.input["query"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text
                    })

            # Tool results go back as a "user" message - this is just the
            # API's convention for "here's the answer to what you asked for."
            conversation_history.append({"role": "user", "content": tool_results})

            response = call_claude(conversation_history)

        answer = get_text_from_response(response)
        print(f"\nAssistant: {answer}\n")

        conversation_history.append({"role": "assistant", "content": response.content})


if __name__ == "__main__":
    run_chat()