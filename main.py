from flask import Flask, render_template, request, session, send_from_directory, jsonify, redirect, url_for
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
import os, time, random
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

AGENT_ROLES = [
    "You are a philosopher. Respond with depth and wisdom.",
    "You are a tech enthusiast. Talk with excitement about new tech.",
    "You are a skeptic. Question everything others say.",
    "You are a peacemaker. Try to keep the conversation balanced and kind."
]

# Agent names
names = ["Aryan", "Varun", "Devansh", "Karan"]

# Global message history
messages = []

def get_video_id(url):
    # For YouTube Shorts, extract the ID from the URL after "shorts/"
    if 'shorts' in url:
        return url.split('/')[-1]
    # For regular YouTube videos, extract the ID after "v="
    elif 'youtube.com' in url and 'v=' in url:
        return url.split('v=')[-1]
    return url  # Default return if no match


@app.route('/')
@app.route('/home')
@app.route('/home.html')
def home():
    feed_items = [
        {'video_url': 'https://www.youtube.com/shorts/4uAz1IVw5F8', 'text': 'Meditate for 10 minutes today'},
        {'video_url': 'https://www.youtube.com/shorts/jZrg1Jqw9uw', 'text': 'Reflect on your actions from yesterday'},
        {'video_url': 'https://www.youtube.com/shorts/iiyMhpV0JaA', 'text': 'Live your Dharma today'},
    ]
    return render_template('index.html', feed_items=feed_items, get_video_id=get_video_id)


@app.route('/guidance', methods=['GET', 'POST'])
@app.route('/guidance.html', methods=['GET', 'POST'])
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
@app.route('/search.html', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form.get('query')
        if query and vectorstore:
            docs = vectorstore.similarity_search(query, k=5)
            results = [doc.page_content for doc in docs]
            return render_template('search.html', results=results, query=query)
    return render_template('search.html')

@app.route('/spiritual')
@app.route('/spiritual.html')
def spiritual():
    # Add your logic here to fetch spiritual wisdom content
    return render_template('spiritual.html')

# Add these routes if they don't exist
@app.route('/games')
@app.route('/games.html')
def games():
    return render_template('games.html')

@app.route('/channels')
@app.route('/channels.html')
def channels():
    channels_data = [
        {
            'name': 'Bhagavad Gita Explained',
            'description': 'Live discourses on the Gita with practical life applications.',
            'image': 'Gallery/The Bhagvad Gita.jpg',
            'live': True
        },
        {
            'name': 'Ramayana Katha',
            'description': 'Heart-touching stories from Ramayana for daily inspiration.',
            'image': 'Gallery/Ramayana.jpg',
            'live': False
        }
    ]
    return render_template('channels.html', channels=channels_data)


@app.route('/community', methods=['GET', 'POST'])
@app.route('/community.html', methods=['GET', 'POST'])
def community():
    if request.method == 'POST':
        user_input = request.form['user_input']
        messages.append(f"You: {user_input}")

        # Last user message is the base for responses
        last_message = user_input

        agents = [llm for _ in range(4)]

        for idx, agent in enumerate(agents):
            system_instruction = AGENT_ROLES[idx]

            # Compose prompt with system instruction and conversation history
            prompt = (
                f"{system_instruction}\n"
                f"Conversation so far:\n{format_history(messages)}\n\n"
                f"Respond to: \"{last_message}\""
            )

            response = agent.invoke([HumanMessage(content=prompt)])

            # Store response as plain text
            messages.append(f"{names[idx]}: {response}")
            last_message = response  # Agents react to previous response

            time.sleep(random.uniform(5, 10))  # Delay between agent replies

    return render_template('community.html', messages=messages)


# Utility function to format history
def format_history(msg_list):
    return "\n".join(msg_list[-10:])  # Only show last 10 messages for context




Daily_challenges = [
    {'id': 1, 'title': '5 min Meditation', 'description': 'Focus fully for 5 minutes.', 'completed': False},
    {'id': 2, 'title': 'Do a Good Deed', 'description': 'Help someone selflessly today.', 'completed': False},
    {'id': 3, 'title': 'Speak Truth', 'description': 'Speak only truth all day.', 'completed': False}
]

@app.route('/challenges', methods=['GET'])
@app.route('/challenges.html', methods=['GET'])
def challenges():
    completed = sum(1 for ch in Daily_challenges if ch['completed'])
    total = len(Daily_challenges)
    remaining = total - completed
    score = completed * 10  # simple scoring rule
    return render_template(
        'challenges.html',
        challenges=Daily_challenges,
        completed=completed,
        remaining=remaining,
        score=score
    )

@app.route('/challenges/complete/<int:ch_id>', methods=['POST'])
def complete_challenge(ch_id):
    for ch in Daily_challenges:
        if ch['id'] == ch_id:
            ch['completed'] = True
            break
    return redirect(url_for('challenges'))

@app.route('/profile')
@app.route('/profile.html')
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