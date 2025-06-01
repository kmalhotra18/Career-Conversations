# 💼 Career Conversations – AI Chatbot by Kunal Malhotra

This is an interactive AI assistant trained on Kunal Malhotra’s professional background and career history. The chatbot uses large language models and Gradio to simulate realistic, high-quality conversations with potential recruiters, clients, or collaborators.

---

## 🚀 Live Demo

Access the deployed chatbot on [Hugging Face Spaces](https://huggingface.co/spaces/kmalhotra18/career_conversations)

---

## 📦 Included Files

- `app.py` — Main Gradio application with streaming chat, tool use, and evaluator integration
- `evaluator.py` — Gemini-powered evaluation module that scores chatbot responses for quality
- `requirements.txt` — List of Python dependencies

> ⚠️ Note: Private profile data (e.g., `me/linkedin.pdf`) is intentionally excluded from the repository.

---

## 🧠 Features

- ✅ OpenAI-compatible LLM interface (GPT-4o or Gemini 1.5)
- ✅ Custom system prompt based on real background
- ✅ Response quality check using Google's Gemini API
- ✅ Tool calls for recording unknown questions and collecting user contact info
- ✅ Modern, responsive Gradio UI

---

## 🛠️ Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/career-conversations.git
   cd career-conversations

2. Create a .env file:
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_gemini_key
PUSHOVER_TOKEN=your_pushover_token (optional)
PUSHOVER_USER=your_pushover_user (optional)

3. Install dependencies:
pip install -r requirements.txt

4. Run the app
