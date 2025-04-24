from flask import Flask, render_template, request, session, send_from_directory, jsonify
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

llm = GoogleGenerativeAI(
      model="gemini-1.5-flash",
      google_api_key="AIzaSyDAtdNAt84hKpdbZR3i2nn-2CEK9typDL8"
  )

# Initialize embedding model and vectorstore
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

try:
    vectorstore = FAISS.load_local(
        'faiss_index_all',
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    print(f"Error loading vectorstore: {e}")
    vectorstore = None

prompt_template = """You are an AI guide and therapist deeply knowledgeable in Dharmic traditions, including Hinduism, Buddhism, Jainism, and Sikhism.
When answering, follow these guidelines:
1. If the question requires historical/philosophical knowledge, retrieve data from FAISS.
2. If the question is broad and opinion-based, generate an answer using LLM.
3. Keep responses *concise yet informative*.
4. Maintain a *scholarly and respectful tone*.

Context:
{context}

User Question: {question}
Answer: """

prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])

memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    input_key="query"
)

@app.route('/')
@app.route('/home')
def home():
    feed_items = [
        {"img": "Mahabharat.jpg", "text": "The Bhagavad Gita teaches us that true wisdom comes from selflessness and devotion."},
        {"img": "Ramayana.jpg", "text": "A tale from the Ramayana: How Hanuman's devotion moved mountains."},
        {"img": "The Bhagvad Gita.jpeg", "text": "Lessons from the Mahabharata: The importance of duty and righteousness."}
    ]
    return render_template('index.html', feed_items=feed_items)

@app.route('/guidance', methods=['GET', 'POST'])
def guidance():
    if 'chat_history' not in session:
        session['chat_history'] = [{
            "role": "assistant",
            "content": "Welcome to DharmaDarshan! How can I help you today?"
        }]

    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if user_input:
            session['chat_history'].append({"role": "user", "content": user_input})
            
            if vectorstore:
  
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    memory=memory,
                    chain_type_kwargs={"prompt": prompt}
                    )
                response = qa_chain({"query": user_input})
                bot_response = response['result']
            else:
                bot_response = "I apologize, but I'm having trouble accessing my knowledge base."
            
            session['chat_history'].append({"role": "assistant", "content": bot_response})
            session.modified = True

    return render_template('guidance.html', chat_history=session.get('chat_history', []))

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form.get('query')
        if query and vectorstore:
            docs = vectorstore.similarity_search(query, k=5)
            results = [doc.page_content for doc in docs]
            return render_template('search.html', results=results, query=query)
    return render_template('search.html')

@app.route('/spiritual')
def spiritual():
    # Add your logic here to fetch spiritual wisdom content
    return render_template('spiritual.html')

# Add these routes if they don't exist
@app.route('/games')
def games():
    return render_template('games.html')

@app.route('/community')
def community():
    return render_template('community.html')

@app.route('/challenges')
def challenges():
    return render_template('challenges.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/Gallery/<path:filename>')
def serve_image(filename):
    return send_from_directory('static/Gallery', filename)

@app.route('/styles.css')
def serve_css():
    return send_from_directory('static', 'styles.css')

if __name__ == '__main__':
    os.makedirs('static/Gallery', exist_ok=True)
    app.run(debug=True)