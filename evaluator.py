from pydantic import BaseModel
import google.generativeai as genai
import os

class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str

def build_system_prompt(name, summary, linkedin):
    return f"""You are an evaluator that decides whether a response to a question is acceptable.
The Agent represents {name} professionally and should respond appropriately.

## Summary:
{summary}

## LinkedIn:
{linkedin}

Evaluate the agent's reply below.
"""

def build_user_prompt(reply, message, history):
    formatted = "\n".join(
        [f"User: {m[0]}\nAgent: {m[1]}" for m in history if isinstance(m, (list, tuple)) and len(m) == 2]
    )
    return f"""Conversation history:\n{formatted}

User's message: {message}

Agent's reply: {reply}

Is this acceptable? Give a short explanation.
"""

def evaluate_reply(name, summary, linkedin, reply, message, history) -> Evaluation:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️ Warning: GOOGLE_API_KEY is missing")
        return Evaluation(is_acceptable=True, feedback="No API key; skipping evaluation")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    system_prompt = build_system_prompt(name, summary, linkedin)
    user_prompt = build_user_prompt(reply, message, history)

    response = model.generate_content([system_prompt, user_prompt])
    content = response.text.strip()

    print("📋 Gemini evaluator response:\n", content)
    is_ok = any(x in content.lower() for x in ["yes", "acceptable", "appropriate", "good"])
    return Evaluation(is_acceptable=is_ok, feedback=content)
