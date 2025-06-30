import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")

# Hugging Face model
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Load the embedding model and FAISS database
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Load LLM from Hugging Face
def load_llm():
    return HuggingFaceHub(
        repo_id=HUGGINGFACE_REPO_ID,
        model_kwargs={"temperature": 0.5, "max_length": 512},
        huggingfacehub_api_token=HF_TOKEN,
    )

# Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
Use the given context to answer the user's question accurately.
If the answer is not in the context, say: "I don't have enough information to answer that."

Context: {context}

Question: {question}

Answer:

"""

# Set the prompt template
def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# Streamlit UI
def main():
    st.title("MediAI")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Ask your question here...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Error loading the FAISS database.")
                return

            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Retrieve more documents
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt()},
            )

            response = qa_chain.invoke({"query": prompt})

            result = response["result"]
            source_documents = response["source_documents"]

            if not result.strip():
                result = "I don't have enough information to answer that."

            result_to_show = f"**Answer:** {result}"#\n\n**Source Docs:**\n{str(source_documents)}"
            
            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
