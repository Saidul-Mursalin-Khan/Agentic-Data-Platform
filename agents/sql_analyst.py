import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from utils.database import DatabaseUtil
from Models.schema import AgentSchema, JudgeSchema
from langchain_core.messages import AIMessage, HumanMessage


# ----------- Ai Agent code -----------


# 3. Curate the questions
def curate_questions(state: AgentSchema) -> AgentSchema:

    user_question = state.user_question

    llm = pick_llm("low")

    response = llm.invoke(f"Curate the following qu estion  : {user_question}").content

    state.curated_question = response
    state.messages = state.messages + [HumanMessage(content=f"{response}")]

    return state


# 4. Curated query + context
def prompt_query_context(state: AgentSchema) -> AgentSchema:
    
    curated_question = state.curated_question

    db_config = {
        "host": os.environ['host'],
        "port": int(os.environ['port']),
        "database": os.environ['database'],
        "user": os.environ['user'],
        "password": os.environ['password'],
    }

    db_obj = DatabaseUtil(db_config)

    schema_info = db_obj.schema_details("public") # fetch schema details for 'public' schema

    # Constructing the prompt query for the agent to generate the SQL query
    prompt = f"""
    You are an SQL analyst agent. Your task is to convert the user's natural language 
    query into Postgres SQL query that can be executed on the database. You are provided 
    with the user's original query and the schema details of the database, including
    table names, column names, data types, and sample data for each table so that 
    you can understand the structure of the database and generate an accurate SQL query.
    Unless user explicitly asks for specific number of rows, always limit the output to 10 rows.
    Note - Just generate the SQL query without any explanation or additional text because
    this query will be executed directly on the database. So, the output should be SQL
    ready to be executed without any modifications.  
    
    User's Original Query: {curated_question}

    Database Schema Details:
    {schema_info}
    
    """    

    state.prompt_query_context = prompt

    return state


# 5. Generate sql query node
def generate_sql(state: AgentSchema) -> AgentSchema:

    prompt = state.prompt_query_context

    llm = pick_llm("medium")

    generated_sql_query = llm.invoke(prompt).content
        
    state.generated_sql_query = generated_sql_query

    return state


# 5. Is safe Node
def is_safe_sql(state: AgentSchema) -> AgentSchema:

    sql_query = state.generated_sql_query;

    llm_obj = pick_llm('medium')
    llm_judge = llm_obj.with_structured_output(JudgeSchema)

    prompt = f"""You are an SQL Judge for data security. Your task is to determine whether the SQL query is safe or not. The SQL query should only be used for data retrieval and should not modify the database in any way. Neither the SQL query nor the prompt should contain any SQL commands that can modify the database, such as INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, or any other commands that can change the structure or content of the database. If the SQL query is safe, respond with "Yes' otherwise respond with
    'No'. Additionally, provide comments explaining your decision. 
    Here is the SQL query to evaluate:
    {sql_query}
    """

    response = llm_judge.invoke(prompt).model_dump()
    state.is_safe_sql_response = response['answer']
    
    return state

# Canceled SQL Query Node
def canceled_sql(state: AgentSchema) -> AgentSchema:

    comments = state.comments

    state.final_answer = f"The generated SQL query was deemed unsafe to execute. The reason provided by the judge is: {comments}. Therefore, the SQL query will not be executed."
    state.messages = state.messages + [AIMessage(content=f"{state.final_answer}")]  # Append the final answer to the messages list  

    return state

# Execute SQL query node
def execute_sql(state: AgentSchema) -> AgentSchema:

    sql_query = state.generated_sql_query

    db_config = {
        "host": os.environ['host'],
        "port": int(os.environ['port']),
        "database": os.environ['database'],
        "user": os.environ['user'],
        "password": os.environ['password'],
    }

    obj = DatabaseUtil(db_config)

    execution_result = obj.execute_sql(sql_query)

    state.sql_query_execution_result = execution_result

    return state


# Represent the final answer node
def represent_final_answer(state: AgentSchema) -> AgentSchema:

    execution_result = state.sql_query_execution_result
    curated_question = state.curated_ques

    llm = pick_llm("low")

    prompt = f"""
    You are an SQL analyst agent. Your task is to provide a final answer to the user based on the
    execution result of the SQL query and the user's original question. The final answer should be
    concise, clear, and directly address the user's query. Avoid including any SQL code or technical
    details in the final answer. The final answer should be in a user-friendly format that is easy to
    understand. If the execution result is empty or does not provide a clear answer to the user's question, explain this in the final answer. \n
    Here is the execution result: {execution_result} \n
    Here is the user's original question: {curated_question}
    """

    response = llm.invoke(prompt).content  # Get the final answer from the LLM

    state.final_answer = response
    state.messages = state.messages + [AIMessage(content=f"{response}")]  # Append the final answer to the messages list

    return state
