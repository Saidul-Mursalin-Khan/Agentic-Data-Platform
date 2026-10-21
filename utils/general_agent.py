from langchain_core.messages import SystemMessage
from utils.llm_pick import pick_llm

GENERAL_PROMPT = """
You are a friendly, helpful assistant inside a Data Agent app.
Answer greetings, small talk and general questions directly and briefly.
If it fits, mention that you can also (1) extract and transform data from APIs
and (2) answer questions about the database in plain English.
"""

def general_node(state):
    llm = pick_llm("medium")
    response = llm.invoke([SystemMessage(content=GENERAL_PROMPT)] + state.messages)
    state.messages = state.messages + [response]
    return state