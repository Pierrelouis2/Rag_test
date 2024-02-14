# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from dotenv import load_dotenv
import os

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_community.llms import Ollama



'''
This function is just to choose which model to use chatgpt API or a local model
'''

def chatGpt(chat_model) :
    #Connect to SQLite database
    db = SQLDatabase.from_uri('sqlite:///chinook.db')

   

    execute_query = QuerySQLDataBaseTool(db=db) # Create a tool to execute the query
    write_query = create_sql_query_chain(chat_model, db) # Create a tool to write the query

# chain = write_query | execute_query # Create a chain to write and execute the query
# response = chain.invoke({"question": "what are the top 3 artist with the most sold tracks?"})
# print(chain.get_prompts())
# print(response)

    answer_prompt = PromptTemplate.from_template(
    """
    Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: 
    
    """
    )
    answer = answer_prompt | chat_model | StrOutputParser()
    chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            result=itemgetter("query") | execute_query
        )
        | answer
    )

    print(chain.invoke({"question": "based on the sold tracks give me the most listened playlist for each countries countries ? (ignore the music playlist )"}))

if __name__ == "__main__": 
    model = input("Enter the model you want to use (chatgpt -> 1 or local ->2 ): ")

    if model == "1" :
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
    
        chat_model = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo") # Create a chat model
        chatGpt(chat_model)
    else :
        chat_model =  Ollama(model="mixtral")
        chatGpt(chat_model)

print("end")
