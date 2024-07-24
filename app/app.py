
# Importing libraries and modules
import streamlit as st
from dotenv import load_dotenv
import logging
import openai
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader, PdfWriter
from tempfile import NamedTemporaryFile
import os
import base64
from htmlTemplates import expander_css, css, bot_interface, user_interface
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Set up Streamlit page configuration
st.set_page_config(layout="wide", 
                   page_title="PDFPal",
                   page_icon=":books:")

# Apply custom CSS
st.write(css, unsafe_allow_html=True)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "N" not in st.session_state:
    st.session_state.N = 0



# Process input PDF
def process_file(doc):
    model_name = "text-embedding-3-large"  
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(api_key=openai_api_key, model=model_name)
    
    pdfsearch = Chroma.from_documents(doc, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.3, api_key=openai_api_key), 
        retriever=pdfsearch.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )
    return chain

# Handle user's query on the uploaded file
def handle_userinput(query):
    response = st.session_state.conversation({"question": query, 'chat_history': st.session_state.chat_history}, return_only_outputs=True)
    st.session_state.chat_history.append((query, response['answer']))

    st.session_state.N = list(response['source_documents'][0])[1][1]['page']

    for i, message in enumerate(st.session_state.chat_history): 
        st.session_state.expander1.write(user_interface.replace("{{MSG}}", message[0]), unsafe_allow_html=True)
        st.session_state.expander1.write(bot_interface.replace("{{MSG}}", message[1]), unsafe_allow_html=True)

def main():
    # Create page layout
    st.session_state.col1, st.session_state.col2 = st.columns([1, 1])
    st.session_state.col1.header("PDFPal :books:")
    user_question = st.session_state.col1.text_input("Ask a question on the uploaded PDF:")
    st.session_state.expander1 = st.session_state.col1.expander('Your Chat', expanded=True)
    st.session_state.col1.markdown(expander_css, unsafe_allow_html=True) 

    # Load and process PDF
    st.session_state.col1.subheader("Your documents")
    st.session_state.pdf_doc = st.session_state.col1.file_uploader("Upload your PDF here and click on 'Process'")

    if st.session_state.col1.button("Process", key='a'):
        with st.spinner("Processing"):
            if st.session_state.pdf_doc is not None:
                with NamedTemporaryFile(suffix=".pdf") as temp:
                    temp.write(st.session_state.pdf_doc.getvalue())
                    temp.seek(0)
                    loader = PyPDFLoader(temp.name)
                    pdf = loader.load()
                    st.session_state.conversation = process_file(pdf)
                    st.session_state.col1.markdown("Done processing. You may now ask a question.")

    # Display reference portion of the document
    if user_question:
        handle_userinput(user_question)
        with NamedTemporaryFile(suffix=".pdf") as temp:
            temp.write(st.session_state.pdf_doc.getvalue())
            temp.seek(0)
            reader = PdfReader(temp.name)
            
            pdf_writer = PdfWriter()
            start = max(st.session_state.N - 2, 0)
            end = min(st.session_state.N + 2, len(reader.pages) - 1) 
            while start <= end:
                pdf_writer.add_page(reader.pages[start])
                start += 1
            with NamedTemporaryFile(suffix=".pdf") as temp2:
                pdf_writer.write(temp2.name)
                with open(temp2.name, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={3}" \
                        width="100%" height="900" type="application/pdf" frameborder="0"></iframe>'
                
                    st.session_state.col2.markdown(pdf_display, unsafe_allow_html=True)

if __name__ == '__main__':
    main()