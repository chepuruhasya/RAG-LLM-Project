import streamlit as st
from rag_pipeline import ask_question

st.title("RAG Document Chatbot")

query = st.text_input("Ask a question about the document")

if query:
    result = ask_question(query)

    st.write("### Answer")
    st.write(result["result"])

    st.write("### Source Documents")

    for doc in result["source_documents"]:
        st.write(doc.page_content)
