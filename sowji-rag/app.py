import os
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import load_prompt
from streamlit import session_state as ss
from pymongo import MongoClient
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import uuid
import json
import time
from langchain.embeddings.openai import OpenAIEmbeddings
import datetime

def is_valid_json(data):
    try:
        json.loads(data)
        return True
    except json.JSONDecodeError:
        return False

# Creating Streamlit title and adding additional information about the bot
st.title("Sowjanya's resumeGPT")
# Display instructions to the user on how to get OpenAI API Key
with st.expander("❓ How to Get Your OpenAI API Key", expanded=False):
    st.markdown("""
### 🔐 Steps to Get OpenAI API Key

**1. Sign up / Sign in:**
- Go to [https://platform.openai.com/signup](https://platform.openai.com/signup)
- Already have an account? Sign in at [https://platform.openai.com/login](https://platform.openai.com/login)

**2. Navigate to API Keys:**
- After logging in, go to [API Keys](https://platform.openai.com/account/api-keys)

**3. Create a New API Key:**
- Click on **“Create new secret key”**
- **Copy the key immediately** and store it securely. You won’t be able to see it again.

**4. Add the API Key to Streamlit:**
- Use environment variable:
```bash
export OPENAI_API_KEY='your-secret-key'

with st.expander("⚠️Disclaimer"):
    st.write("""This is a work in progress chatbot based on a large language model. It can answer questions about Sowjanya""")

# Get OpenAI API key from user
openai_api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()

# Check if the API key is valid by trying to create embeddings
try:
    test_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    test_embeddings.embed_query("test")
    st.sidebar.success("API key is valid!")
except Exception as e:
    st.sidebar.error("Invalid API key. Please check your key and try again.")
    st.stop()

path = os.path.dirname(__file__)

# Loading prompt to query openai
prompt_template = path+"/templates/template.json"
prompt = load_prompt(prompt_template)

# loading embedings
faiss_index = path+"/faiss_index"

# Loading CSV file
data_source = path+"/data/about_me.csv"
pdf_source = path+"/data/AI Ml Sowjanya.pdf"

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

#using FAISS as a vector DB
if os.path.exists(faiss_index):
    vectors = FAISS.load_local(faiss_index, embeddings, allow_dangerous_deserialization=True)
else:
    # Creating embeddings for the docs
    if data_source:
        # Load data from PDF and CSV sources
        pdf_loader = PyPDFLoader(pdf_source)
        pdf_data = pdf_loader.load_and_split()
        print(pdf_data)
        csv_loader = CSVLoader(file_path=data_source, encoding="utf-8")
        csv_data = csv_loader.load()
        data = pdf_data + csv_data
        vectors = FAISS.from_documents(data, embeddings)
        vectors.save_local("faiss_index")

retriever = vectors.as_retriever(search_type="similarity", search_kwargs={"k":6, "include_metadata":True, "score_threshold":0.6})

def get_conversation_chain():
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            temperature=0.0,
            model_name='gpt-3.5-turbo', 
            openai_api_key=openai_api_key
        ), 
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        chain_type="stuff",
        max_tokens_limit=4097, 
        combine_docs_chain_kwargs={"prompt": prompt}
    )

def conversational_chat(query):
    with st.spinner("Thinking..."):
        try:
            chain = get_conversation_chain()
            result = chain({
                "system": "You are a Art's ResumeGPT chatbot, a comprehensive, interactive resource for exploring Sowjanya (Sowjanya) Bojja 's background, skills, and expertise. Be polite and provide answers based on the provided context only. Use only the provided data and not prior knowledge.", 
                "question": query, 
                "chat_history": st.session_state['history']
            })
        except Exception as e:
            st.error("API error occurred. Please check your API key and try again.")
            print(e)
            return "I encountered an error while processing your request. Please try again."
    
    if (is_valid_json(result["answer"])):              
        data = json.loads(result["answer"])
    else:
        data = json.loads('{"answered":"false", "response":"Hmm... Something is not right. I\'m experiencing technical difficulties. Try asking your question again or ask another question about Sowjanya\'s professional background and qualifications. Thank you for your understanding.", "questions":["What is Sowjanya\'s professional experience?","What projects has Sowjanya worked on?","What are Sowjanya\'s career goals?"]}')
    
    answered = data.get("answered")
    response = data.get("response")
    questions = data.get("questions")

    st.session_state['history'].append((query, response))
    
    if ('I am tuned to only answer questions' in response) or (response == ""):
        full_response = """Unfortunately, I can't answer this question. My capabilities are limited to providing information about Sowjanya Bojja's professional background and qualifications. If you have other inquiries, I recommend reaching out to Sowjanya on [LinkedIn](https://www.linkedin.com/in/sowjanya-bojja/). I can answer questions like: \n - What is Sowjanya's educational background? \n - Can you list Sowjanya's professional experience? \n - What skills does Sowjanya possess? \n"""
    else: 
        markdown_list = ""
        for item in questions:
            markdown_list += f"- {item}\n"
        full_response = response + "\n\n What else would you like to know about Sowjanya? You can ask me: \n" + markdown_list
    
    return full_response

if "uuid" not in st.session_state:
    st.session_state["uuid"] = str(uuid.uuid4())

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        welcome_message = """
            Welcome! I'm **Sowjanya's ResumeGPT**, specialized in providing information about Sowjanya's professional background and qualifications. Feel free to ask me questions such as:

            - What is Sowjanya's educational background?
            - Can you outline Sowjanya's professional experience?
            - What skills and expertise does Sowjanya bring to the table?

            I'm here to assist you. What would you like to know?
            """
        message_placeholder.markdown(welcome_message)

if 'history' not in st.session_state:
    st.session_state['history'] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about Sowjanya"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        user_input = prompt
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = conversational_chat(user_input)
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
