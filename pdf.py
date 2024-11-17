from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import re  # Importing the regular expression module

def main():
    load_dotenv()

    st.set_page_config(page_title="Ask your PDF")    # Configures the Streamlit page with a specific title
    st.header("Ask your PDF 💬")

    # Uploading file
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    # Custom check for file size
    max_size = 1024 * 1024 * 1024  # 1GB in bytes
    if uploaded_file is not None and uploaded_file.size > max_size:
        st.warning(f"File size exceeds 1GB limit. Please upload a file smaller than 1GB.")
        return

    # Ask for a valid question after uploading the PDF
    if uploaded_file is not None:
        st.write(uploaded_file)
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split input data into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,  # characters
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        # Create embeddings -> encode information for effective search
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Create input field for asking questions
        user_question = st.text_input("Ask a question about PDF")

        # Check if the user question is alphabetic
        if user_question == "" or not re.match(r'^[a-zA-Z\s]+$', user_question):
            st.warning("Please provide a valid question.")
            return

        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI()  # Large language model - OpenAI in our case
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            # If no answer found, provide a message and Google link
            if not response or response.strip().lower() in ["", "i don't know"]:
                st.warning("I don't know the answer to that.")
                st.markdown(f"[Search on Google for further information](https://www.google.com/search?q={user_question.replace(' ', '+')})")
            else:
                st.write(response)

if __name__ == "__main__":
    main()
