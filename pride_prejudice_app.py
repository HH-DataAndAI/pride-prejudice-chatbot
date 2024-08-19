from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
# from hf_token.py import HuggingFaceAPIToken

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# llm
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=hf_model)

# document
loader = PyPDFLoader("pride_and_prejudice.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                               chunk_overlap=150)
docs = text_splitter.split_documents(documents)

# embeddings
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = "cache/"
os.makedirs(embeddings_folder, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                   cache_folder=embeddings_folder)

# creating vector database
# vector_db = FAISS.from_documents(docs, embeddings)
# vector_db.save_local("/content/faiss_index")

# load Vector Database
# allow_dangerous_deserialization is needed. Pickle files can be modified to deliver a malicious payload that results in execution of arbitrary code on your machine
vector_db = FAISS.load_local("/content/faiss_index", embeddings, allow_dangerous_deserialization=True)

# retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# memory
@st.cache_resource
def init_memory(_llm):
    return ConversationBufferMemory(
        llm=llm,
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)
memory = init_memory(llm)

# prompt
template = """You are a nice chatbot having a conversation with a human. Answer the question based only on the following context and previous conversation. Keep your answers short and succinct.

Previous conversation:
{chat_history}

Context to answer question:
{context}

New human question: {question}
Response:"""

prompt = PromptTemplate(template=template,
                        input_variables=["context", "question"])

# chain
chain = ConversationalRetrievalChain.from_llm(llm,
                                              retriever=retriever,
                                              memory=memory,
                                              return_source_documents=True,
                                              combine_docs_chain_kwargs={"prompt": prompt})


##### streamlit #####

st.title("Pride & Prejudice: conversations in early 19th-century England")

# Initialise chat history
# Chat history saves the previous messages to be displayed
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Witty minds welcomed!"):

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Wandering through the halls of Pemberley for answers..."):

        # send question to chain to get answer
        answer = chain(prompt)

        # extract answer from dictionary returned by chain
        response = answer["answer"]

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer["answer"])

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
