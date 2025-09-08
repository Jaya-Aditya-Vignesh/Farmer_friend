import os
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/api/chat", methods=['POST'])
def chat_with_assistant():
    data = request.get_json()
    user_message = data.get('message')

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        system_prompt = (
            "You are a knowledgeable AI Farmer Assistant based in India. Your goal is to provide "
            "clear, concise, and practical advice to farmers. Cover topics like crop management, "
            "pest control, soil health, and weather adaptation, considering Indian climate and conditions."
        )

        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        ai_response = completion.choices[0].message.content
        return jsonify({"reply": ai_response})

    except Exception as e:
        return jsonify({"error": f"An error occurred with the AI service: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)