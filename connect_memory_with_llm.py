import os
from dotenv import load_dotenv, find_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")

# Safe model: publicly available, compatible with Hugging Face Inference API
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Load LLM
def load_llm():
    return HuggingFaceHub(
        repo_id=HUGGINGFACE_REPO_ID,
        model_kwargs={"temperature": 0.5, "max_length": 512},
        huggingfacehub_api_token=HF_TOKEN,
    )

# Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}

Question: {question}

Answer:

"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Embeddings + FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
)

# Run query
user_query = input("Write your query here: ")
response = qa_chain.invoke({"query": user_query})
print("\nRESULT:\n", response["result"])
#print("\nSOURCE DOCUMENTS:\n", response["source_documents"])
