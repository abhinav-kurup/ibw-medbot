from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Qdrant
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from qdrant_client import QdrantClient

app = Flask(__name__)

load_dotenv()
QDRANT_PATH = os.getenv("QDRANT_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

qa = None

def load_llm():
    global qa

    embeddings = download_hugging_face_embeddings()

    client = QdrantClient(path=QDRANT_PATH)


    docsearch = Qdrant(
        client=client, 
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )


    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain_type_kwargs = {"prompt": PROMPT}


    llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                        model_type="llama",
                        config={'max_new_tokens':512,
                                'temperature':0.8})


    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True, 
        chain_type_kwargs=chain_type_kwargs)

load_llm()

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        input = msg
        print(f"Received message: {input}")
        try:
            result = qa({"query": input})
            print("QA Chain response:", result["result"])
            return str(result["result"])
        except Exception as qa_error:
            print(f"Error in QA Chain: {qa_error}")
            return f"Error processing your query: {str(qa_error)}", 500
    
    except Exception as e:
        print(f"Unexpected error in chat route: {e}")
        return "An unexpected error occurred", 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)