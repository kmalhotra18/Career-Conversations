from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
from evaluator import evaluate_reply  # import evaluator

# Load environment variables
load_dotenv(override=True)

# Push notification helpers
def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

# Tool JSON definitions
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.name = "Kunal Malhotra"

        # Load the LinkedIn and summary
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    # Handle tool calls
    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    # System prompt
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def stream_chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        collected = ""
        done = False

        while not done:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
                stream=True
            )
            for chunk in response:
                choice = chunk.choices[0]
                delta = choice.delta

                if choice.finish_reason == "tool_calls":
                    message_obj = choice.message
                    tool_calls = message_obj.tool_calls
                    tool_results = self.handle_tool_call(tool_calls)
                    messages.append(message_obj)
                    messages.extend(tool_results)
                    return  # skip evaluation for tool responses

                if delta and delta.content:
                    token = delta.content
                    collected += token
                    yield collected

                if choice.finish_reason == "stop":
                    done = True

        # Evaluation
        evaluation = evaluate_reply(self.name, self.summary, self.linkedin, collected, message, history)
        if evaluation.is_acceptable:
            print("✅ Passed evaluation")
            yield collected
        else:
            print("❌ Failed evaluation. Retrying...")
            print(f"📝 Feedback: {evaluation.feedback}")
            updated_system = self.system_prompt() + f"\n\n## Previous answer rejected\nResponse: {collected}\nReason: {evaluation.feedback}"
            retry_messages = [{"role": "system", "content": updated_system}] + history + [{"role": "user", "content": message}]
            retry = self.openai.chat.completions.create(model="gpt-4o-mini", messages=retry_messages)
            yield retry.choices[0].message.content

    
# --- Gradio UI ---
def launch_gradio(me: Me):
    with gr.Blocks(css="""
    .gr-chatbot {
        border-radius: 12px !important;
        border: 1px solid #e0e0e0;
        background-color: #fdfdfd;
        padding: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .gr-textbox textarea {
        border-radius: 8px !important;
        border: 1px solid #ccc !important;
        padding: 10px !important;
        font-size: 16px;
    }
    .gr-button {
        border-radius: 8px !important;
        font-weight: bold;
    }
    """, title="Ask Kunal") as demo:

        with gr.Row():
            with gr.Column(scale=1, min_width=80):
                gr.Image("https://ui-avatars.com/api/?name=Kunal+Malhotra&size=128", show_label=False, container=False)
            with gr.Column(scale=9):
                gr.Markdown("""
                    # 👋 Ask Kunal Malhotra  
                    Welcome to my interactive assistant. Curious about my background, skills, or experience? Ask me anything below!
                """)

        chatbot = gr.Chatbot(label="", type="messages", elem_classes="gr-chatbot")
        state = gr.State([])

        with gr.Row():
            msg = gr.Textbox(placeholder="Type your question here...", scale=8, lines=2, autofocus=True)
            submit = gr.Button("🚀 Send", scale=1)
            clear = gr.Button("🧹 Clear", scale=1)

        def handle_submit(user_input, chat_history):
            chat_history = chat_history or []
            response_gen = me.stream_chat(user_input, chat_history)
            collected = ""
            for delta in response_gen:
                collected = delta
                yield chat_history + [{"role": "user", "content": user_input}, {"role": "assistant", "content": collected}], \
                      chat_history + [{"role": "user", "content": user_input}, {"role": "assistant", "content": collected}]

        msg.submit(handle_submit, [msg, state], [chatbot, state])
        submit.click(handle_submit, [msg, state], [chatbot, state])
        clear.click(lambda: ([], []), None, [chatbot, state])

    demo.launch()
  
    

if __name__ == "__main__":
    me = Me()
    launch_gradio(me)
    