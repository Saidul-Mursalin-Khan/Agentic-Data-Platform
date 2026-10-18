import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from Models.schema import AgentSchema

llm_obj = pick_llm('low')
print(llm_obj.invoke("Where is Dhaka?").content)

# ----------- Ai Agent code -----------

# 3. Curate the questions
def curate_questions(state: AgentSchema) -> AgentSchema:
    user_question = state.user_question
    llm = pick_llm("low")
    response = llm.invoke(f"Curate the following qu estion  : {user_question}")
    state.curated_questions = response
    return state

def prompt_query_context(state: AgentSchema) -> AgentSchema:
    curate_question = state.curated_questions