import os
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.chains import RetrievalQA

# Initialize models
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version="2023-05-01-preview",
    azure_deployment="gpt-4",
)

embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version="2023-05-15",
    azure_deployment="text-embedding-ada-002",
)

# Test the LLM model
response = llm.invoke("tell me a joke about Programming")
print("LLM output:", response)

# Load documents
doc_loaders = [
    PyPDFLoader("src/document/financial_doc.pdf"),
    WebBaseLoader("https://www.langchain.com/langsmith")
]
documents = [documents.extend(loader.load()) for loader in doc_loaders]
print(documents[0])

# Chunk the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Create a vector store
vector_store = FAISS.from_documents(chunks, embedding_model)

# Define the RAG pipeline
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Function to process query
def process_query(query, n=3):
    result = rag_chain({"query": query})
    response = result["result"]
    retriever = vector_store.as_retriever(search_kwargs={"k": n})
    docs = retriever.get_relevant_documents(query)
    top_docs = [(doc.page_content, doc.metadata.get("score", 0.0)) for doc in docs]
    return top_docs, response

# Test queries
test_queries = [
    "What are the key components of a langsmith?",
    "What is the difference between RAG and Fine-tuning?"
]

# Save to CSV
import csv
with open("src/FinalResponse.csv", "w", newline="") as csvfile:
    fieldnames = ["top_docs", "response"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for query in test_queries:
        top_docs, response = process_query(query, n=3)
        writer.writerow({"top_docs": str(top_docs), "response": response})
        print(f"Query: {query}")
        print(f"Top documents: {top_docs}")
        print(f"Response: {response}\n")