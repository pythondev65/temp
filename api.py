from flask import Flask, request, jsonify
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
import os

app = Flask(__name__)

def save_response_to_file(response):
    with open("data/data.txt", "a") as file:
        file.write(response + '\n')

def create_gpt_output(question, temperature=0.01, key=None):
   loader = DirectoryLoader('data', glob='**/*.txt', show_progress=True, loader_cls=TextLoader)
   documents = loader.load()
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
   texts = text_splitter.split_documents(documents)
   embeddings = OpenAIEmbeddings(openai_api_key=key)
   docsearch = FAISS.from_documents(texts, embeddings)
   retriever = docsearch.as_retriever()
   llm = OpenAI(openai_api_key=key, temperature=temperature)
   qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
   query = str(question)
   result = qa({"query": query})
   print("result : ", result)
   return result['result']


# API endpoint to handle POST requests
@app.route('/generate_response', methods=['POST'])
def generate_response():
    try:
        # Get the input data from the request
        data = request.get_json()
        question = data.get('question')
        prompt_txt = "Add Google Cloud SSML tags to the following text. Include a <speak> tag at the beginning of the paragraph and <break> tags for pauses. Address any missing commas and periods. Optimize the content for narration without adding or altering any additional text. Ensure that the final result accurately represents the original paragraph with the added SSML tags: "
        final_question = f"{prompt_txt} {question}"
        temperature = 0.01
        openai_api_key = data.get('openai_api_key')

        # Validate OpenAI API key
        if not openai_api_key:
            return jsonify({"error": "OpenAI API key is required"}), 400

        # Call the existing function
        result = create_gpt_output(final_question, temperature, openai_api_key)



        # Return the result as JSON
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save_prompt', methods=['POST'])
def save_prompt():
    try:
        # Get the input data from the request
        data = request.get_json()
        question = data.get('prompt')

        save_response_to_file(question)

        # Return the result as JSON
        return jsonify({"result": "saved"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)