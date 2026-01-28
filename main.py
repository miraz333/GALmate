import os
import base64
import asyncio
import edge_tts
import nest_asyncio
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- APPLY ASYNCIO FIX ---
nest_asyncio.apply()

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
# Security: Key is now ONLY loaded from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found in environment variables.")

model = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.7
)

sessions_history = {}

@app.route('/')
def home():
    return "LAGmate Server is Running. Connect via your frontend."

# --- VOICE GENERATION ---
async def generate_voice_base64(text):
    try:
        voice = "en-US-AriaNeural"
        communicate = edge_tts.Communicate(text, voice)
        
        temp_filename = f"temp_{os.urandom(8).hex()}.mp3"
        await communicate.save(temp_filename)
        
        with open(temp_filename, "rb") as audio_file:
            encoded_string = base64.b64encode(audio_file.read()).decode("utf-8")
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        return encoded_string
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

# --- CHAT ENDPOINT ---
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("text", "")
    session_id = data.get("session_id", "default_user")
    
    default_instruction = (
        "You are LAGmate, a professional English Tutor. "
        "1. Speak ONLY in English. "
        "2. Help the user practice English conversation. "
        "3. Correct their grammar gently if needed. "
        "4. Keep answers concise (2-3 sentences)."
    )
    
    system_instruction = data.get("system_instruction", default_instruction)

    if not user_input:
        return jsonify({"error": "No text"}), 400

    if session_id not in sessions_history:
        sessions_history[session_id] = []

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{topic}")
        ])

        formatted_prompt = prompt.invoke({
            "topic": user_input,
            "chat_history": sessions_history[session_id]
        })
        
        result = model.invoke(formatted_prompt)
        ai_reply = result.content

        sessions_history[session_id].append(HumanMessage(content=user_input))
        sessions_history[session_id].append(AIMessage(content=ai_reply))
        
        if len(sessions_history[session_id]) > 10:
            sessions_history[session_id] = sessions_history[session_id][-10:]

        try:
            audio_data = asyncio.run(generate_voice_base64(ai_reply))
        except Exception as audio_err:
            audio_data = None

        return jsonify({
            "reply": ai_reply,
            "audio": audio_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
