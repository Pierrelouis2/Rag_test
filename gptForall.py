# import openai

# openai.api_base = "http://localhost:4891/v1"
# #openai.api_base = "https://api.openai.com/v1"

# openai.api_key = ""

# # Set up the prompt and other parameters for the API request
# prompt = "Who is Michael Jordan?"

# # model = "gpt-3.5-turbo"
# #model = "mpt-7b-chat"
# model = "mistral-7b-instruct-v0.1.Q4_0"

# # Make the API request
# response = openai.Completion.create(
#     model=model,
#     prompt=prompt,
#     max_tokens=50,
#     temperature=0.28,
#     top_p=0.95,
#     n=1,
#     echo=True,
#     stream=False
# )

# # Print the generated completion
# print(response)
# from gpt4all import GPT4All
# model = GPT4All("mistral-7b-instuct-v0.1.Q4_0.gguf")
# with model.chat_session():
#     response1 = model.generate(prompt='hello', temp=0)
#     response2 = model.generate(prompt='write me a short poem', temp=0)
#     response3 = model.generate(prompt='thank you', temp=0)
#     print(model.current_chat_session)
import time
import sqlite3

from gpt4all import GPT4All
start_time = time.time()
model = GPT4All("C:/Users/TelepP/.cache/gpt4all/mistral-7b-instruct-v0.1.Q4_0.gguf",device="Radeon (TM) RX 480 Graphics")
# output = model.generate("what is the capital of france", max_tokens=10)
with model.chat_session():
    while 1:
        text = input("Enter your text: ")
        if text == "exit":
            break
        response = model.generate(prompt=text, temp=0.5, max_tokens=100)
        print(response)
    # time_response_1 = time.time()
    # response1 = model.generate(prompt='hello', temp=0)
    # time_response_2 = time.time()
    # response2 = model.generate(prompt='write me a short poem', temp=0)
    # time_response_3 = time.time()
    # response3 = model.generate(prompt='thank you', temp=0)
    # print(model.current_chat_session)
# print(output)
# print(f"""
#     Time to respond to hello: {time_response_2 - time_response_1}
#     Time to respond to write me a short poem: {time_response_3 - time_response_2}
#     Time to respond to thank you: {time.time() - time_response_3}
#     Time to load the model: {time_response_1 - start_time}
# """)


# conn = sqlite3.connect('chinook.db')

# # Create a cursor
# cursor = conn.cursor()

# # # Execute a query
# # cursor.execute("""
# #   select * from sqlite_master where type='table';
# # """)

# cursor.execute("""
#   select type, name, tbl_name from sqlite_master where type='table';
# """)
# # Fetch all rows from the last executed statement
# rows = cursor.fetchall()

# question = "what is the most listened track"


# template = f"""
#     You are a SQLite expert. You are given the db schema :
#     {rows}
#     You will have to write a SQL query to answer the following question:
#     {question}

#     Use the following format for the output:
#     SQLQuery: SQL Query to run

#     """

# response = model.generate(prompt=template, temp=0, max_tokens=100)
# print(response)
print("--- %s seconds ---" % (time.time() - start_time))