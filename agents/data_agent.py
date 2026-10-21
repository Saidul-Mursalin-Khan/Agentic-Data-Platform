import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from utils.etl_tools import ETLTools
from Models.schema import RouterSchema, DataAgentSchema
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from agents.etl_analyst import etl_analyst
from agents.sql_analyst import sql_analyst


llm = pick_llm("medium")

llm_router = llm.with_structured_output(RouterSchema)


# ---------------------------- PROMPTS ---------------------------- #

ROUTER_PROMPT = """
You are a router for a Data Agent. Read the user's message and choose ONE route:

- "etl": the user wants to extract, load or transform data from an API or a file
  (e.g. fetch data from an endpoint, save it as csv/json/parquet, filter a file, clean data).
- "sql": the user asks a question that must be answered from the ride-sharing database
  (tables: users, vehicles, rides, payments, ratings), e.g. counts, averages, lists, trends.
- "general": everything else: greetings, small talk, general knowledge, questions about
  what you can do, explanations of concepts, or anything unrelated to the two routes above.

When unsure between "general" and the other routes, choose "general".
Return only the route name in the `answer` field.
"""

GENERAL_PROMPT = """
You are a friendly, helpful assistant inside a Data Agent app.
Answer greetings, small talk and general questions directly and briefly.
If it fits the conversation, you may mention that you can also:
1) extract and transform data from APIs or files (ETL), and
2) answer questions about the ride-sharing database in plain English (SQL).
"""


# ---------------------------- DATA AGENT GRAPH ---------------------------- #

def router_node(state:DataAgentSchema):

    message = state.messages[-1].content

    route_response_dict = llm_router.invoke(
        [SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=message)]
    ).model_dump()

    route_response = str(route_response_dict['answer']).strip().lower()

    state.route_response = route_response

    return state

def etl_node(state:DataAgentSchema):

    message = state.messages[-1].content

    response = etl_analyst.invoke(
             {"messages":[HumanMessage(content=f"""
            {message}
    """)]}
        ) 

    # etl_analyst returns its whole state (a dict). Keep only its last written answer
    # as a normal AIMessage, so the last item always has a readable `.content`.
    answer = next(
        (m for m in reversed(response["messages"])
         if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None)),
        None,
    )
    if answer is None:
        answer = AIMessage(content="The ETL agent finished, but did not return a message.")

    state.messages = state.messages + [answer]

    return state

def sql_node(state: DataAgentSchema):

    message = state.messages[-1].content

    input_schema = {
        "messages": [HumanMessage(content=message)],
        "user_question": f"{message}",
        "curated_question": "",
        "prompt_query_context": "",
        "generated_sql_query": "",
        "is_safe": "no",
        "comments": "",
        "sql_query_execution_result": "",
        "final_answer": ""
    }

    response = sql_analyst.invoke(input_schema)

    state.messages = state.messages + [response]

    return state

def general_node(state: DataAgentSchema):
    """Answers greetings and generic questions directly with the LLM."""

    # Only real Human/AI messages with text go to the LLM (SQL results in history are dicts, so they are skipped)
    history = [
        m for m in state.messages
        if isinstance(m, (HumanMessage, AIMessage)) and m.content
    ]

    response = llm.invoke([SystemMessage(content=GENERAL_PROMPT)] + history[-10:])

    state.messages = state.messages + [AIMessage(content=response.content)]

    return state

data_agent_graph = StateGraph(DataAgentSchema)

data_agent_graph.add_node("router_node", router_node)
data_agent_graph.add_node("etl_node", etl_node)
data_agent_graph.add_node("sql_node", sql_node)
data_agent_graph.add_node("general_node", general_node)

data_agent_graph.add_edge(START, "router_node")

def route_edge(state: DataAgentSchema) -> str:
    if state.route_response == "sql":
        return "sql_node"
    elif state.route_response == "etl":
        return "etl_node"
    else:
        # "general" or anything unexpected: chat with the LLM instead of crashing
        return "general_node"


data_agent_graph.add_conditional_edges("router_node", route_edge,
                                      {
                                          "sql_node": "sql_node",
                                          "etl_node": "etl_node",
                                          "general_node": "general_node"
                                      })

# The sql_node and etl_node end the graph (same as before). The general_node ends it too.
data_agent_graph.add_edge("general_node", END)

data_agent = data_agent_graph.compile()


if __name__ == "__main__":

    # Optional: draw the graph (runs only when this file is executed directly, not when the CLI imports it)
    try:
        from IPython.display import display, Image
        img = Image(data_agent.get_graph().draw_mermaid_png())
        with open("data_agent_graph.png", "wb") as f:
            f.write(img.data)
    except Exception as e:
        print(f"Graph image skipped: {e}")

    for question in ["hello", "What is a data lake?", "How many payment method are available in the database?"]:
        response = data_agent.invoke(
            {"messages":[HumanMessage(content=question)],
             "route_response": ""}
        )

        last_item = response["messages"][-1]

        if isinstance(last_item, dict):
            print(last_item.get("final_answer", "No final answer found"))
        else:
            print(last_item.content)