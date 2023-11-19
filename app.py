import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import time
# get a token: https://platform.openai.com/account/api-keys

from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.chains import RetrievalQAWithSourcesChain

from PIL import Image
import base64
import wikipedia
st.set_page_config(
    page_title="Wikipedia QnA",
    page_icon=Image.open("logo.png"),
    layout="wide",
)
app_heading_css = """
    <style>
        .container {
            display: flex;
        }
        .logo-text {
            font-weight:700 !important;
            font-size:40px !important;
        }
        .logo-img {
            float:right;
            margin-right:20px !important;
        }
    </style>
    """
app_heading_html = app_heading_css + f"""
    <div class="container">
        <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open("logo.png", "rb").read()).decode()}" width=70 height=70>
        <p class="logo-text">{"Wikipedia QnA"}</p>
    </div>
    """
st.markdown(
    app_heading_html,
    unsafe_allow_html=True
)


 
os.environ["OPENAI_API_KEY"] = "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"

 
def main():
    st.header("Chat with PDF 💬")
 
 
    # get link
    input = st.text_input("Enter your url")
    urls=[input]
    if input is not None and input != "":
        loaders = UnstructuredURLLoader(urls=urls)
        data = loaders.load()
        # Text Splitter
        text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=1000,
                                      chunk_overlap=200)
        docs = text_splitter.split_documents(data)
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_documents(docs, embeddings)
 
 
        # Initialize chats
        if "allChats" not in st.session_state:
            st.session_state.allChats = [[]]
            st.session_state.currentChat = []
            st.session_state.chatIndex = 0
            


        with st.sidebar:
            #Create a button to increment and display the number
            if st.button('New chat+'):
                st.session_state.allChats.append([])
                st.session_state.currentChat=st.session_state.allChats[len(st.session_state.allChats)-1]
                st.session_state.chatIndex=len(st.session_state.allChats)-1
            for chat in range(len(st.session_state.allChats)):
                #button_key = f'button_{i}'
                if st.button(f'chat-{chat+1}'):
                    st.session_state.currentChat= st.session_state.allChats[chat]
                    st.session_state.chatIndex=chat
            add_vertical_space(5)
            st.write('Made with ❤️ by Ajeer')

        wiki_URL = input
        if wiki_URL is not None and wiki_URL != "":
            st.components.v1.iframe(src=wiki_URL, width=None, height=550, scrolling=True)

        
            


        # Display chat messages from history on app rerun
        for message in st.session_state.currentChat:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("What is up?"):
            # Add user message to chat history
            st.session_state.currentChat.append({"role": "user", "content": prompt})
            st.session_state.allChats[st.session_state.chatIndex]=st.session_state.currentChat
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                docs = VectorStore.similarity_search(query=prompt, k=3)
                llm = OpenAI()
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())
                response=chain({"question": prompt}, return_only_outputs=True)
                assistant_response=response['answer']

                # Simulate stream of response with milliseconds delay
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            # Add assistant response to chat history
            st.session_state.currentChat.append({"role": "assistant", "content": full_response})
 
if __name__ == '__main__':
    main()