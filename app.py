import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

arxiv_wrapper = ArxivAPIWrapper(doc_content_chars_max=250, top_k_results=1)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(doc_content_chars_max=250, top_k_results=1)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchResults(name="search")

st.title("ChatBot with Tools and Agents.")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key: ", type="password")

if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": "Hi, I'm a ChatBot wo can search anything on the internet, how can i help you? "}
    ]

for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:= st.chat_input(placeholder="what is deep learning"):
    st.session_state.messages.append({'role': "user", "content": prompt})
    st.chat_message('user').write(prompt)

    llm = ChatGroq(model='gemma2-9b-it', api_key=api_key, streaming=True, max_tokens=1000)
    tools = [search, wiki, arxiv]

    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_error=True)

    with st.chat_message('assistant'):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)