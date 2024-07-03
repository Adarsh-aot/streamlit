import streamlit as st
import csv
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from langchain.llms import HuggingFacePipeline


chroma_client = chromadb.PersistentClient(path="data_db")


sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")


collection = chroma_client.get_or_create_collection(name="my_collection", embedding_function=sentence_transformer_ef)


# Streamlit app layout
st.title("ChromaDB and HuggingFace Pipeline Integration")

query = st.text_input("Enter your query:", value="director")

if st.button("Search"):
    results = collection.query(
        query_texts=[query],
        n_results=3,
        include=['documents', 'distances', 'metadatas']
    )
    st.write("Query Results:")
    st.write(results['metadatas'])

    if results['documents']:
        context = results['documents'][0][0]
        st.write("Context:")
        st.write(context)
        tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-T5-738M")
        model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-T5-738M")

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512
        )

        local_llm = HuggingFacePipeline(pipeline=pipe)

        l = f"""
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {query}
        Helpful Answer:
        """

        answer = local_llm(l)
        st.write("Answer:")
        st.write(answer)    

    
            