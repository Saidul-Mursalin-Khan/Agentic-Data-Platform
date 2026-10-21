import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from utils.etl_tools import ETLTools
from Models.schema import ETLAgentSchema
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
# from langchain_anthropic import ChatAnthropic

# --------------------- ETL AGENT TOOLS ------------------------

@tool
def extract_load_tool(url:str, output_folder:str, format:str, query_params:str = "", file_name:str = "") -> str:
    """
    This tool extracts the data from the API (url) and loads it into the
    the desired location (output_folder).

    Args:
        url (str): The API endpoint from which to extract data, WITHOUT query string.
        output_folder (str): The folder where the extracted data will be saved.
        format (str): The format in which to save the extracted data (csv, json, parquet).
        query_params (str): Optional server-side filters as a query string,
            e.g. "tag=programming&per_page=100" for the dev.to articles API.
            Leave empty when no filtering is requested.
        file_name (str): A short, simple file name WITHOUT extension that describes the data,
            in lowercase with underscores, 2-4 words, e.g. "pokemon_list" or "devto_programming_posts".
            Existing files are never overwritten (a number is added if the name is taken).
    
    Returns:
        str: A message indicating the success or failure of the operation.

    """
    etl_tools = ETLTools()
    return etl_tools.extract_load(url, output_folder, format, query_params, file_name)


@tool
def transform_load_tool(input_file_path:str, output_folder:str, output_format:str, user_question:str, file_name:str = "") -> str:
    """
    This tool transforms the data from the specified file and loads it into the
    desired location (output_folder).

    Args:
        input_file_path (str): The path to the file containing the data to be transformed.
            Use the exact path returned by extract_load_tool when it was just run.
            A folder is also accepted: the newest file inside it is used.
        output_folder (str): The folder where the transformed data will be saved.
        output_format (str): The format in which to save the transformed data (csv, json, parquet).
        user_question (str): The transformation to perform, in plain words
            (e.g. "keep only rows where tags contain programming").
        file_name (str): A short, simple file name WITHOUT extension that describes the result,
            in lowercase with underscores, 2-4 words, e.g. "bulbasaur_only" or "python_posts".
            Existing files are never overwritten (a number is added if the name is taken).
    
    Returns:
        str: A message indicating the success or failure of the operation.

    """
    etl_tools = ETLTools()

    df = etl_tools.read_file(input_file_path)
    top_3_rows = etl_tools.transform_load_context(input_file_path)

    llm = pick_llm("medium")

    # The LLM only TRANSFORMS. Reading and saving are done by ETLTools,
    # so unsupported disk commands never appear in generated code.
    prompt = f"""
            You are a Python Data Analyst who uses Pandas to analyze data. 
            A Pandas DataFrame named `df` is ALREADY loaded from the file : {input_file_path}
            Write only the Pandas code that transforms `df` as per the user's question
            and assigns the final DataFrame to a variable named `result_df`.

            Rules:
            - Output only Python code. No explanation, no comments, no markdown.
            - Do NOT import anything. `pd` (pandas) and `np` (numpy) are already available.
            - Do NOT read or write any file (no read_csv, to_csv, to_json, to_parquet, open, os).
              Saving is done for you.
            - List-like columns (such as tags) are stored as comma-separated strings, so filter
              them with df[col].astype(str).str.contains("value", case=False, na=False).

            Here's the user's question: {user_question}\n
            Here's the context of the data you will be analyzing: {top_3_rows}\n
        """

    # Try up to 3 times, feeding the error back to the LLM
    error_note = ""
    result_df = None
    pandas_code = ""
    last_error = ""

    for _ in range(3):
        response = llm.invoke(prompt + error_note).content

        pandas_code = etl_tools.clean_code(response)

        try:
            result_df = etl_tools.execute_code(pandas_code, df)
            break
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            error_note = (
                f"\n\nYour previous code failed.\nCode:\n{pandas_code}\nError: {last_error}\n"
                f"Fix it and return only the corrected code."
            )

    if result_df is None:
        return f"Transformation failed after 3 attempts. Last error: {last_error}\n\n Last code tried: \n {pandas_code}"

    saved_path = etl_tools.save_dataframe(result_df, output_folder, file_name or "transformed_data", output_format)

    return (
        f"The data is transformed and saved at {saved_path} in {output_format} format. "
        f"Rows after transformation: {len(result_df)}. \n\n Pandas Code Executed: \n {pandas_code} \n\n "
        f"Execution Result (first rows): \n {result_df.head(5).to_string()}"
    )


# Toolkit 
tools = [extract_load_tool, transform_load_tool]

llm = pick_llm("medium")
llm_bind = llm.bind_tools(tools)


# --------------------- ETL AGENT GRAPH ------------------------

SYSTEM_PROMPT = """
            You are a Python Data Analyst who has access to tools that can extract and load, 
            transform and load data. You will be provided with a user's question 
            and you would need to perform the right ETL operations as per the user's question. 

            Guidelines:
            - If the user wants only to extract data, call extract_load_tool once.
            - If the user wants filtered data from an API that supports filters, pass the filter
              through `query_params` (for dev.to articles: "tag=programming"). The url stays clean:
              'https://dev.to/api/articles'.
            - If the filter cannot be done by the API, extract first, then call transform_load_tool
              on the extracted file.
            - "csv folder" or "in csv" means the csv format, not a folder name.
            - If a tool reports an error, fix the arguments and try again, or explain the problem.
            If the operation is performed then inform the user and end the coversation.
"""

def sanitize_messages(messages):
    """
    OpenAI rejects a history where an assistant message with tool_calls is not
    IMMEDIATELY followed by one ToolMessage per tool_call_id (error 400).

    This rebuilds the history so that rule always holds:
    - every tool call gets its ToolMessage right after it (a placeholder if the tool never answered)
    - ToolMessages without a matching tool call are dropped
    - anything that is not a LangChain message (e.g. a stray dict) is dropped
    """
    replies = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}

    cleaned = []
    for m in messages:
        if not isinstance(m, BaseMessage) or isinstance(m, ToolMessage):
            continue   # tool replies are re-inserted right after their tool call below

        cleaned.append(m)

        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                reply = replies.get(tc["id"]) or ToolMessage(
                    content="Tool call did not finish.", tool_call_id=tc["id"]
                )
                cleaned.append(reply)

    return cleaned

def llm_node(state:ETLAgentSchema):

    messages = state.messages

    # Real message objects (not a string dump) so tool calls and tool results stay linked
    final_answer = llm_bind.invoke([SystemMessage(content=SYSTEM_PROMPT)] + sanitize_messages(messages))

    state.messages = messages + [final_answer]

    return state

def tool_node(state:ETLAgentSchema):
    """
    This node is responsible for invoking the appropriate tool based on the user's question and the context provided by the LLM.
    """

    tools_results = []

    tools_by_name = {tool.name: tool for tool in tools}

    tool_calls = state.messages[-1].tool_calls

    for tool_call in tool_calls:

        # Every tool call MUST get a ToolMessage, even when the tool name is wrong or the tool fails.
        # A failing tool must not crash the graph: send the error back to the LLM instead
        try:
            tool = tools_by_name[tool_call['name']]
            observation = tool.invoke(tool_call['args'])
        except Exception as e:
            observation = f"Tool failed: {type(e).__name__}: {e}"

        tools_results.append(ToolMessage(content=str(observation), tool_call_id = tool_call['id']))

    state.messages = state.messages + tools_results

    return state   

# Nodes and Edges
etl_analyst_graph = StateGraph(ETLAgentSchema)
etl_analyst_graph.add_node("llm_node", llm_node)
etl_analyst_graph.add_node("tool_node", tool_node)

etl_analyst_graph.add_edge(START, "llm_node")

def is_tool_call(state:ETLAgentSchema):
    tool_calls = getattr(state.messages[-1], "tool_calls", None)

    if tool_calls:
        return "tool_node"
    else:
        return "end"

etl_analyst_graph.add_conditional_edges(
    "llm_node",is_tool_call,
    {
        "tool_node": "tool_node",
        "end": END
    }
)

etl_analyst_graph.add_edge("tool_node", "llm_node")

etl_analyst = etl_analyst_graph.compile()


if __name__ == "__main__":
    llm_bind = pick_llm("medium").bind_tools(tools)
    print(llm_bind.invoke("I want to extract the data from the api endpoint 'https://dev.to/api/articles'"))

    # Optional
    from IPython.display import display, Image
    img = Image(etl_analyst.get_graph().draw_mermaid_png())
    with open("etl_analyst_graph.png", "wb") as f:
        f.write(img.data)

    response = etl_analyst.invoke(
        {"messages":[HumanMessage(content="I want to extract the data from the API endpoint 'https://dev.to/api/articles', filter the posts by programming tag and save it to data/extract folder in csv format")]}
    )

    response = etl_analyst.invoke(
         {"messages":[HumanMessage(content=f"""
            I want to transform the data stored in the '\\Users\\saidulmursalinkhan\\Developer\\Agentic_Data_Platform\\data\\extract' file 
            and save the transformed data in the '\\Users\\saidulmursalinkhan\\Developer\\Agentic_Data_Platform\\data\\transform' folder in the csv format.
            The transformation should keep only the posts whose tags contain python.
            """)]
        }
    )

    print(response)