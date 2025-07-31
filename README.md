DharmaDarshan

DharmaDarshan is a transformative platform designed to nurture your personal and spiritual growth through a wide range of immersive experiences. From community support and daily dharma challenges to motivational feeds and guidance from spiritual gurus — DharmaDarshan is your companion on the path to a more mindful, fulfilling life.

🌟 Key Features

🔹 Community Support: Connect with like-minded individuals in dedicated communities to share experiences and grow together.

🔹 Personal Dharma Guider: Get personalized guidance for real-life challenges using principles from Dharma and ancient wisdom.

🔹 Mindful Mini-Games: Engage with mentally stimulating and spiritually uplifting games designed to improve clarity and focus.

🔹 Motivational Content Feed: A home page that keeps your spirit energized with:
         Inspirational quotes      Mythological reels         Dharma-based life lessons       Spiritual stories

🔹 Daily Dharma Challenges: Stay on track with daily tasks to keep your energy aligned and your spirit motivated.

🔹 Live & Recorded Guru Sessions: Follow your favorite gurus and motivational influencers through live and recorded spiritual talks and classes.

🔹 Follow Channels: Subscribe to topic-specific channels for curated content and events.

🛠️ Tech Stack

Python
FAISS for semantic search
Flask
HTML/CSS/JS for frontend templates
Static asset system (static/, templates/)

Custom model logic in model.py

📁 Project Structure
bash
Copy
Edit
/static/                 # Static assets (JS, CSS, images)
templates/              # Front-end HTML templates
faiss_index_all/        # Pre-computed FAISS indices
main.py                 # App entry point
model.py                # Core search and logic handler
requirements.txt        # Python dependencies

⚙️ Setup Instructions

Clone the repository
git clone https://github.com/Aryan556gaur/DharmaDarshan.git
cd DharmaDarshan

Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

Install dependencies
pip install -r requirements.txt

Run the app
python main.py

Then visit: http://localhost:5000

🤝 Contributing
We welcome contributions that align with the mission of DharmaDarshan.

Fork this repository

Create your feature branch (git checkout -b feature/new-feature)

Commit your changes (git commit -m 'Add new feature')

Push to the branch (git push origin feature/new-feature)

Open a Pull Request

📄 License
This project is licensed under the MIT License.

Let Dharma guide your steps — one day, one lesson, one challenge at a time. 🙏

⚠️ Make sure the FAISS index files are correctly placed in faiss_index_all/.
