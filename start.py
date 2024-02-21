# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from dotenv import load_dotenv
import os

from langchain_community.llms import (Ollama,GPT4All,LlamaCpp)
from langchain_core.prompts import PromptTemplate
'''
This function is just to choose which model to use chatgpt API or a local model
'''

def chatGpt(chat_model) :
    #Connect to SQLite database
    db = SQLDatabase.from_uri('sqlite:///chinook.db')

    template = """
    You are a SQLite expert.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per SQLite.
    Use the following format:
    SQLQuery: SQL Query to run
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    Only use the following tables:{table_info}
    Question: {input}
    """

   
    prompt = PromptTemplate.from_template(template)
    print("prompt",prompt)
    execute_query = QuerySQLDataBaseTool(db=db) # Create a tool to execute the query
    write_query = create_sql_query_chain(chat_model, db,prompt=prompt) # Create a tool to write the query
    # print("write_query",write_query)
    chain = write_query | execute_query # Create a chain to write and execute the query
    # config = {
    #     'max_tokens':  2048  # Set the maximum token limit here
    # }
    response = chain.invoke({"question": "what are the top 3 artist with the most sold tracks?"})
    print(chain.get_prompts())
    print(response)

    # answer_prompt = PromptTemplate.from_template(
    # """
    # Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    # Question: {question}
    # SQL Query: {query}
    # SQL Result: {result}
    # Answer: 
    
    # """
    # )
    # answer = answer_prompt | chat_model | StrOutputParser()
    # chain = (
    #     RunnablePassthrough.assign(query=write_query).assign(
    #         result=itemgetter("query") | execute_query
    #     )
    #     | answer
    # )

    # print(chain.invoke({"question": "based on the sold tracks give me the most listened playlist for each countries countries ? (ignore the music playlist )"}))

if __name__ == "__main__": 
    model = input("Enter the model you want to use (chatgpt -> 1 or local ->2 ): ")

    if model == "1" :
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
    
        chat_model = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo") # Create a chat model
        chatGpt(chat_model)
    else :
        # chat_model = LlamaCpp(model_path="C:/Users/TelepP/.cache/gpt4all/llama-2-7b-chat.Q2_K.gguf", n_ctx=4096)
        chat_model = GPT4All(model="C:/Users/TelepP/.cache/gpt4all/llama-2-7b-chat.Q2_K.gguf",device="Radeon (TM) RX 480 Graphics",)
        # chat_model =  Ollama(model="dolphin-phi")
        chatGpt(chat_model)

print("end")
