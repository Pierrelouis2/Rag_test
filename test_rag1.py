import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import (OllamaEmbeddings,GPT4AllEmbeddings)
from gpt4all import GPT4All


# Constants
PDF_FILE_PATH = './t.pdf'  # Update with your file path
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
OLLAMA_MODEL_NAME ="C:/Users/TelepP/.cache/gpt4all/mistral-7b-instruct-v0.1.Q4_0.gguf"

# Function to load and split the PDF document
def load_and_split_pdf(file_path):
    """
    Loads a PDF document and splits it into pages.
    Args:
    file_path (str): Path to the PDF file.
    Returns:
    list: List of pages from the PDF document.
    """
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# Function to split text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    """
    Splits text from pages into smaller chunks.
    Args:
    pages (list): List of pages from the document.
    chunk_size (int): Size of each text chunk.
    chunk_overlap (int): Overlap between consecutive chunks.
    Returns:
    list: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(pages)

# Function to create embeddings and vector store
def create_embeddings_and_vectorstore(splits, model_name):
    """
    Creates embeddings and a vector store from document splits.
    Args:
    splits (list): List of document splits.
    model_name (str): Name of the Ollama model to use.
    Returns:
    Chroma: Vector store object.
    """
    # embeddings = OllamaEmbeddings(model=model_name)
    embeddings = GPT4AllEmbeddings(model=model_name,device="Radeon (TM) RX 480 Graphics")
    return Chroma.from_documents(documents=splits, embedding=embeddings)

# Function to format documents for context
def format_docs(docs):
    """
    Formats a list of documents into a single string.
    Args:
    docs (list): List of documents.
    Returns:
    str: Formatted string of document contents.
    """
    return "\n\n".join(doc.page_content for doc in docs)

# Function to query Ollama LLM with context
def ollama_llm(question, context):
    """
    Queries the Ollama LLM with a question and context.
    Args:
    question (str): The question to ask.
    context (str): The context for the question.
    Returns:
    str: The response from the LLM.
    """
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    print(formatted_prompt)
    # chat_model = LlamaCpp(model_path="C:/Users/TelepP/.cache/gpt4all/llama-2-7b-chat.Q2_K.gguf", n_ctx=4096)
    response = GPT4All(OLLAMA_MODEL_NAME,device="Radeon (TM) RX 480 Graphics").generate(prompt=formatted_prompt, temp=0.5, max_tokens=100)
    # response = ollama.chat(model=OLLAMA_MODEL_NAME, messages=[{'role': 'user', 'content': formatted_prompt}])
    print("---------------------------first test------------------------")
    print(response)
    print("---------------------------End first test------------------------")
    print("---------------------------second test------------------------")
    second_promt = f"Question: {question} \n The answer is : {response} \n Make it more clear and simple to understand in less than 100 words"
    print("---------------------------second Prompt------------------------")
    print(second_promt)
    print("---------------------------EnD second Prompt------------------------")
    response2 = GPT4All(OLLAMA_MODEL_NAME,device="Radeon (TM) RX 480 Graphics").generate(prompt=second_promt, temp=0.5, max_tokens=100)
    print(response2)
    print("---------------------------End Second test------------------------")
    # return response['message']['content']
    return response
# Function to execute the RAG chain
def rag_chain(question, retriever):
    """
    Executes the RAG chain to get an answer to a question.
    Args:
    question (str): The question to ask.
    retriever: The retriever object for document retrieval.
    Returns:
    str: The answer to the question.
    """
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

# Main Execution Flow
def main():
    """
    Main function to execute the RAG workflow.
    """
    print("Loading and splitting PDF...")
    pages = load_and_split_pdf(PDF_FILE_PATH)
    print("Creating embeddings and vector store...")
    splits = split_text(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    print("Creating retriever...")
    vectorstore = create_embeddings_and_vectorstore(splits, OLLAMA_MODEL_NAME)
    retriever = vectorstore.as_retriever()
    print("Executing example queries...")
    # Example Queries
    query1 = "What are the 4 steps to create the Guru IA ?"
    result1 = rag_chain(query1, retriever)
    # print("Result 1:", result1)

    print("End of execution.")

    # query2 = "What were the key highlights and financial performance of Nvidia's Data Center segment in the third quarter of fiscal year 2024?"
    # result2 = rag_chain(query2, retriever)
    # print("Result 2:", result2)

if __name__ == "__main__":
    main()