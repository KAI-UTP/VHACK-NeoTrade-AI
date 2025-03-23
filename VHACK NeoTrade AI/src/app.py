from flask import Flask, render_template, request, jsonify
from google.generativeai import configure, GenerativeModel

# ✅ Set your Gemini API key properly
configure(api_key="AIzaSyDN_Ct7X-pAr2Lrq8ogHIQSo-dQHgFa6Dc")

# Pre-load model instance
model = GenerativeModel("gemini-2.0-flash")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/policy')
def policy():
    return render_template("policy.html")

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    try:
        # Provide system-level context about the web app
        system_message = (
            "You are an AI chatbot assistant for the web app 'NeoTrade AI'. "
            "The site helps users explore AI-powered crypto trading signals and insights. "
            "It includes sections like Home, Policy, Tutorial, Media, Trading, and Log in / Sign in. "
            "Your goal is to assist users by answering questions about how the app works and how to navigate it."
            "When responding to user questions, try to break down steps into short, readable points. Use bullet points or numbered lists, and keep language simple and direct."
            "Answer with simple, like in one or two sentences only"
            )

        response = model.generate_content([system_message, user_input])
        reply = response.text.strip() if hasattr(response, "text") else "No response received."
        return jsonify({"reply": reply})
    except Exception as e:
        print("Gemini API Error:", e)
        return jsonify({"reply": "Sorry, there was an error generating a response."})

if __name__ == '__main__':
    print("Running app...")
    app.run(debug=True)
