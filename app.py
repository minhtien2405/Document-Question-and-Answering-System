import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Qdrant
# from langchain.vectorstores import FAISS
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template

import warnings
warnings.filterwarnings("ignore", category=FutureWarning )


def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    splitter = CharacterTextSplitter(
        separator = '\n',
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    text_chunks = splitter.split_text(raw_text)
    return text_chunks

def get_vector_store(text_chunks):
    '''We can use Cohere-embed-multilingual-v3.0 in https://huggingface.co/cohere/embed-multilingual-v3.0 
    embeddings = HuggingFaceEmbeddings(model_name='cohere/embed-multilingual-v3.0')
    and vector_store = Qdrant(embeddings=embeddings)
    '''
    # embeddings = OpenAIEmbeddings(openai_api_key = os.getenv('OPENAI_API_KEY'))
    embeddings = HuggingFaceInstructEmbeddings(model_name = 'hkunlp/instructor-xl')
    # vector_store = FAISS.from_texts(texts = text_chunks, embedding = embeddings)    
    url = 'http://localhost:6333'
    vector_store = Qdrant.from_texts(texts = text_chunks, embedding = embeddings, url = url)
    return vector_store

def get_conversation_chain(vector_store):
    llm = HuggingFaceHub(repo_id = 'google/flan-t5-xxl', model_kwargs = {'temperature': 0.5, 'max_length': 512}) # temperature is the randomness of the model
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    # st.write(response)
    
    '''Example of response
    {
        "question":"WHAT IS PERSON'S NAME IN CV"
        "chat_history":[
            0:"HumanMessage(content="WHAT IS PERSON'S NAME IN CV")"
            1:"AIMessage(content='Pham Minh Tien')"
        ]
        "answer":"Pham Minh Tien"
    }
    '''
    
    st.session_state.chat_history = response['chat_history']
    for index, message in enumerate(st.session_state.chat_history):
        if index % 2 == 0:
            st.write(user_template.replace('{{MSG}}', message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace('{{MSG}}', message.content), unsafe_allow_html=True)   

def main():
    load_dotenv() 
    st.set_page_config(page_title='Chat with multiple Documents', 
                       page_icon= ':books:',
                       layout='wide', 
                       initial_sidebar_state='auto')
    
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('Chat with multiple Documents :books:')
    user_question = st.text_input('Ask a question about the documents:')
    if user_question:
        if st.session_state.conversation:
            handle_user_input(user_question)
        else:
            st.warning('Please upload your documents and click on "Process" to start the conversation.', icon = "⚠️")

    # st.write(user_template.replace('{{MSG}}', user_question), unsafe_allow_html=True)
    st.write(bot_template.replace('{{MSG}}', 'Hi, I am your virtual assistant. I can help you with your questions about the documents. Please upload your documents and click on "Process" to start the conversation.'), unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader('Your Documents')
        pdf_docs = st.file_uploader(
            "Upload your documents here and click on 'Process':",
            type=['pdf', 'docx'],
            accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner('Processing your documents...'):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)
                # create vector store
                vector_store = get_vector_store(text_chunks)
                # create conversation
                st.session_state.conversation = get_conversation_chain(vector_store)
                st.info('Your documents have been processed successfully!', icon='🚨')
                





if __name__ == '__main__':
    main()
