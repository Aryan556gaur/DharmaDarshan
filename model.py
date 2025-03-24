import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Streamlit App Title
st.title("💬 Psychiatrist and Spiritual Expert Chatbot")
st.write("Ask me anything about psychiatry or spirituality. Type 'exit' to stop.")

# Step 1: Load FAISS Vector Store
@st.cache_resource  # Cache the embedding model and vector store to avoid reloading
def load_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local('faiss_index_all', embeddings=embedding_model, allow_dangerous_deserialization=True)
    return vector_store.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 relevant chunks

retriever = load_vector_store()

# Step 2: Initialize Google Gemini 1.5 Flash
@st.cache_resource  # Cache the LLM to avoid reinitialization
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="AIzaSyDAtdNAt84hKpdbZR3i2nn-2CEK9typDL8")

llm = load_llm()

# Step 3: Define the RAG Prompt
rag_prompt = PromptTemplate(
    template="You are a psychiatrist as well as a spiritual expert. Use the following passages to answer the question.\n\nContext:\n{context}\n\nConversation History:\n{chat_history}\n\nQuestion: {question}\nAnswer:",
    input_variables=["context", "chat_history", "question"]
)

# Step 4: Initialize Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Step 5: Create the Conversational Retrieval QA Chain
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": rag_prompt}
    )

# Step 6: Chatbot Interface
user_input = st.text_input("You: ", key="input")

if user_input:
    if user_input.lower() == "exit":
        st.write("🤖 Chatbot: Goodbye!")
        st.stop()
    else:
        # Invoke the chain
        result = st.session_state.qa_chain.invoke({"question": user_input})
        st.write(f"🤖 Chatbot: {result['answer']}")