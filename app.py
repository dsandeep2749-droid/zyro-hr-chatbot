
import os
from glob import glob
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


st.set_page_config(page_title="Zyro HR Help Desk", page_icon="💼")

st.title("💼 Zyro Dynamics HR Help Desk")


@st.cache_resource
def build_rag():

    from glob import glob

pdf_paths = glob("*.pdf")

documents = []

    for pdf in pdf_paths:
        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15}
    )

    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0,
        max_tokens=512
    )

    prompt = ChatPromptTemplate.from_template("""
    You are an HR Help Desk assistant for Zyro Dynamics.

    RULES:
    1. Answer ONLY from context
    2. If answer not found say:
       "I can only answer questions related to Zyro Dynamics HR policies."
    3. Do not hallucinate

    Context:
    {context}

    Question:
    {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)[:3000]

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, vectorstore


chain, vectorstore = build_rag()

SIMILARITY_THRESHOLD = 1.2

query = st.chat_input("Ask your HR question...")


if query:

    with st.chat_message("user"):
        st.write(query)

    docs_and_scores = vectorstore.similarity_search_with_score(query, k=3)

    if len(docs_and_scores) == 0 or docs_and_scores[0][1] > SIMILARITY_THRESHOLD:
        response = "I can only answer questions related to Zyro Dynamics HR policies."
        sources = []
    else:
        response = chain.invoke(query)
        sources = docs_and_scores

    with st.chat_message("assistant"):
        st.write(response)

        if sources:
            st.subheader("📚 Sources")
            for doc, score in sources:
                st.write(
                    f"File: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}"
                )
