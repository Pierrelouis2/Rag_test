from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import GPT4All
import time

start_time = time.time()
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

local_path = 'C:/Users/TelepP/.cache/gpt4all/mistral-7b-instruct-v0.1.Q4_0.gguf'  # replace with your desired local file path

# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]

# Verbose is required to pass to the callback manager
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)

# If you want to use a custom model add the backend parameter
# Check https://docs.gpt4all.io/gpt4all_python.html for supported backends
llm = GPT4All(model=local_path, backend="gptj", callbacks=callbacks, verbose=True,device="Radeon (TM) RX 480 Graphics")

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_chain.run(question)

print("--- %s seconds ---" % (time.time() - start_time))