"""
compliance_app.py
Streamlit UI for the tool-calling compliance agent (compliance_agent.py).

This follows the same pattern as your earlier app.py (the Streamlit version
of sop_chat.py), with one new piece: the tool-calling loop. In the terminal
version, that loop was a plain `while` loop inside run_chat(). Here, it has
to live inside the function that handles a single user message, because
Streamlit reruns the whole script on every interaction - there's no long-
running "loop" the way a terminal script has. Everything that needs to
persist across those reruns (conversation history, the vector store) goes
into st.session_state / @st.cache_resource, exactly like before.
"""

import os
import streamlit as st
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


# @st.cache_resource makes Streamlit build the vector store ONCE, the first
# time the app runs, and reuse it on every rerun after that - instead of
# rebuilding it (reloading the PDF, re-embedding 57 chunks) every single
# time you send a message, which is what would happen without this.
@st.cache_resource
def get_vector_store():
    return build_vector_store()


def execute_sop_search(vector_store, query):
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
    return "".join(block.text for block in response.content if block.type == "text")


def get_assistant_reply(vector_store, conversation_history, status_placeholder):
    """
    This is the Streamlit version of the tool-use while-loop from
    compliance_agent.py's run_chat(). Same exact logic - call Claude, check
    if it asked for search_sops, run it, feed the result back, repeat until
    it's done. The only new thing is status_placeholder, which shows a
    small "searching SOPs for: ..." message in the UI while it works,
    instead of printing to a terminal.
    """
    response = call_claude(conversation_history)

    while response.stop_reason == "tool_use":
        conversation_history.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "search_sops":
                status_placeholder.markdown(f"🔍 *Searching SOPs for: {block.input['query']}*")
                result_text = execute_sop_search(vector_store, block.input["query"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text
                })

        conversation_history.append({"role": "user", "content": tool_results})
        response = call_claude(conversation_history)

    status_placeholder.empty()
    return response


st.set_page_config(page_title="Compliance Assistant", page_icon="🛡️")


def check_password():
    """
    A simple gate: shows a password box, and only lets the rest of the app
    render once the correct password is entered. This checks against
    st.secrets, the same secure secrets store where your ANTHROPIC_API_KEY
    lives once deployed - the password itself never sits in your code or
    your GitHub repo, only in Streamlit Cloud's secrets settings (or a
    local .streamlit/secrets.toml file for testing on your own machine).

    st.session_state remembers that this person already got in, so it
    doesn't re-ask on every single interaction - only once per browser
    session.
    """
    if st.session_state.get("password_correct", False):
        return True

    st.title("🛡️ Compliance Assistant")
    entered = st.text_input("Enter password to continue", type="password")

    if entered:
        if entered == st.secrets.get("APP_PASSWORD"):
            st.session_state.password_correct = True
            st.rerun()  # immediately re-run the script so the real app renders below
        else:
            st.error("Incorrect password.")

    return False


if not check_password():
    st.stop()  # halts execution here - nothing below this line runs until the password is right


st.title("🛡️ Compliance Assistant")
st.caption("Ask general SOP questions, or get help reviewing a GSR, ZTL, or business/website case.")

vector_store = get_vector_store()

# st.session_state persists across reruns - a plain Python variable here
# would get wiped and recreated every time you type a message, since
# Streamlit re-runs the entire script top to bottom on every interaction.
if "history" not in st.session_state:
    st.session_state.history = []       # what gets sent to Claude (includes tool blocks)
if "display_history" not in st.session_state:
    st.session_state.display_history = []  # what gets shown on screen (clean Q&A only)

# Redraw the conversation so far on every rerun
for msg in st.session_state.display_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question or describe a case...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.display_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        with st.spinner("Thinking..."):
            response = get_assistant_reply(vector_store, st.session_state.history, status_placeholder)
            answer = get_text_from_response(response)
        st.markdown(answer)

    st.session_state.history.append({"role": "assistant", "content": response.content})
    st.session_state.display_history.append({"role": "assistant", "content": answer})