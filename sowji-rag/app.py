import os
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import load_prompt
from streamlit import session_state as ss
import uuid
import json
import datetime

def is_valid_json(data):
    try:
        json.loads(data)
        return True
    except json.JSONDecodeError:
        return False

# Creating Streamlit title and adding additional information about the bot
st.title("Sowjanya's resumeGPT")
with st.expander("⚠️Disclaimer"):
    st.write("""This is a work in progress chatbot based on a large language model. It can answer questions about Sowjanya""")
st.markdown("""
### How to get an OpenAI API Key

To use this chatbot, you'll need an OpenAI API key. Here's how to get one:

1. **Go to [OpenAI Platform](https://platform.openai.com/)**
   - Sign up or log in to your account

2. **Access API Keys**
   - Click your profile icon (top-right)
   - Select "View API keys"

3. **Create New Key**
   - Click "Create new secret key"
   - Name it (e.g., "ResumeGPT")
   - Copy the key (it starts with `sk-` and won't be shown again!)

4. **Enter it below** ⬇️
   - Paste in the sidebar input box
   - The app will verify it automatically

⚠️ **Important Notes:**
- Keys are sensitive - don't share them!
- Free tier has usage limits
- You may need to add payment method for continued use
""")
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
    st.sidebar.error(f"Invalid API key. Error: {str(e)}")
    st.stop()

path = os.path.dirname(__file__)

# Loading prompt to query openai
prompt_template = os.path.join(path, "templates/template.json")
try:
    prompt = load_prompt(prompt_template)
except Exception as e:
    st.error(f"Failed to load prompt template: {str(e)}")
    st.stop()

# loading embeddings
faiss_index = os.path.join(path, "faiss_index")

# Loading data files
data_source = os.path.join(path, "data/about_me.csv")
pdf_source = os.path.join(path, "data/AI Ml Sowjanya.pdf")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Using FAISS as a vector DB
if os.path.exists(faiss_index):
    try:
        vectors = FAISS.load_local(faiss_index, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed to load FAISS index: {str(e)}")
        st.stop()
else:
    # Creating embeddings for the docs
    if data_source and pdf_source:
        try:
            # Load data from PDF and CSV sources
            pdf_loader = PyPDFLoader(pdf_source)
            pdf_data = pdf_loader.load_and_split()
            
            csv_loader = CSVLoader(file_path=data_source, encoding="utf-8")
            csv_data = csv_loader.load()
            
            data = pdf_data + csv_data
            vectors = FAISS.from_documents(data, embeddings)
            vectors.save_local("faiss_index")
        except Exception as e:
            st.error(f"Failed to create embeddings: {str(e)}")
            st.stop()

retriever = vectors.as_retriever(
    search_type="similarity", 
    search_kwargs={"k":6, "include_metadata":True, "score_threshold":0.6}
)

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
                "system": "You are a ResumeGPT chatbot, a comprehensive resource for exploring Sowjanya Bojja's background, skills, and expertise. Be polite and provide answers based on the provided context only.", 
                "question": query, 
                "chat_history": st.session_state['history']
            })
        except Exception as e:
            st.error(f"API error occurred: {str(e)}")
            return "I encountered an error while processing your request. Please try again."
    
    try:
        if is_valid_json(result["answer"]):              
            data = json.loads(result["answer"])
        else:
            data = {
                "answered": "false", 
                "response": "Hmm... Something is not right. I'm experiencing technical difficulties. Try asking your question again.",
                "questions": [
                    "What is Sowjanya's professional experience?",
                    "What projects has Sowjanya worked on?",
                    "What are Sowjanya's career goals?"
                ]
            }
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")
        data = {
            "answered": "false",
            "response": "Sorry, I encountered an error processing the response.",
            "questions": []
        }

    answered = data.get("answered", "false")
    response = data.get("response", "")
    questions = data.get("questions", [])

    st.session_state['history'].append((query, response))
    
    if ('I am tuned to only answer questions' in response) or not response:
        full_response = """Unfortunately, I can't answer this question. My capabilities are limited to providing information about Sowjanya Bojja's professional background. I can answer questions like: 
        - What is Sowjanya's educational background? 
        - Can you list Sowjanya's professional experience? 
        - What skills does Sowjanya possess?"""
    else: 
        markdown_list = "\n".join([f"- {item}" for item in questions])
        full_response = f"{response}\n\nWhat else would you like to know about Sowjanya? You can ask me:\n{markdown_list}"
    
    return full_response

# Initialize session state
if "uuid" not in st.session_state:
    st.session_state["uuid"] = str(uuid.uuid4())

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []
    with st.chat_message("assistant"):
        welcome_message = """
            Welcome! I'm **Sowjanya's ResumeGPT**, specialized in providing information about Sowjanya's professional background. Feel free to ask me questions such as:

            - What is Sowjanya's educational background?
            - Can you outline Sowjanya's professional experience?
            - What skills does Sowjanya have?

            I'm here to assist you. What would you like to know?
            """
        st.markdown(welcome_message)

if 'history' not in st.session_state:
    st.session_state['history'] = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about Sowjanya"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_response = conversational_chat(prompt)
        st.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
