import os
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_dance.contrib.google import make_google_blueprint, google
from google import genai
from google.genai import types
from flask import redirect, url_for
from flask_dance.consumer import oauth_authorized
from flask import flash
from sqlalchemy.orm.exc import NoResultFound
from flask import session
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_login import LoginManager, login_required, current_user, UserMixin, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
import asyncio
import nest_asyncio
import joblib
from data_fetcher import create_features
from cnn_feature_extractor import load_model as load_cnn
from hmm_regime_detector import load_hmm
from hybrid_model import load_model as load_hybrid
from signal_generator import generate_signals
from live_data_fetcher import data_current_df
from live_trade import signals
from datetime import datetime
from flask_socketio import SocketIO, emit
from live_trade import last_model_update


app = Flask(__name__)                     # ✅ app is defined here
socketio = SocketIO(app, cors_allowed_origins="*")  # ✅ now it's safe to use here

cnn_model = tf.keras.models.load_model('saved_models/cnn.keras')
hybrid_model = tf.keras.models.load_model('saved_models/hybrid.keras')
hmm_model = joblib.load('saved_models/hmm.pkl')

# ML Input
# This will store the latest 22 prices
rolling_data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
price_window = []

def preprocess(latest_price):
    global price_window

    price_window.append(float(latest_price))
    if len(price_window) > 22:
        price_window = price_window[-22:]

    if len(price_window) < 22:
        padded = [0.0] * (22 - len(price_window)) + price_window
    else:
        padded = price_window

    # Repeat price 14 times (or use [price, 0, 0, ...] if only price matters)
    features = [[price] + [0]*13 for price in padded]

    return np.array(features).reshape(1, 22, 14)


# Import the db, Post, and Comment from your separate models.py
from models import db, Post, Comment

# Allow nested event loops (for async calls inside Flask)
nest_asyncio.apply()

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Example secret key

# -------------------
# (1) Configure Database for community.db
# -------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///community.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Bind the SQLAlchemy db to this app
db.init_app(app)

# Create tables if they don't exist yet
with app.app_context():
    db.create_all()

config = types.GenerateContentConfig(
    temperature=0.7,
    top_k=40,
    top_p=0.95,
    max_output_tokens=1024,
    response_mime_type="text/plain"
)

# Initialize Gemini client with your API key
client = genai.Client(api_key="AIzaSyDN_Ct7X-pAr2Lrq8ogHIQSo-dQHgFa6Dc")

# Setup Google OAuth
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # For development only

google_bp = make_google_blueprint(
    client_id="780519547787-5fjg6tetlqgqboc3hcmprsoq4edn9kbf.apps.googleusercontent.com",
    client_secret="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",
    scope=[
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "openid"
    ],
    redirect_url="/login/google/authorized"
)

app.register_blueprint(google_bp, url_prefix="/login")

# User session management
login_manager = LoginManager()
login_manager.login_view = "google.login"
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, id, name, email, profile_pic):
        self.id = id
        self.name = name
        self.email = email
        self.profile_pic = profile_pic

users = {}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

@app.route("/login")
def custom_login():
    session["next_url"] = request.args.get("next") or url_for("home")
    return redirect(url_for("google.login"))

@oauth_authorized.connect_via(google_bp)
def google_logged_in(blueprint, token):
    if not token:
        flash("Failed to log in with Google.", category="error")
        return False

    # Get user info
    resp = blueprint.session.get("/oauth2/v2/userinfo")
    if not resp.ok:
        flash("Failed to fetch user info from Google.", category="error")
        return False

    user_info = resp.json()
    user_id = user_info["id"]
    user = User(id=user_id, name=user_info["name"], email=user_info["email"], profile_pic=user_info["picture"] )
    users[user_id] = user
    login_user(user)

    flash("Successfully signed in with Google.", category="success")

    # Redirect manually using session "next_url"
    next_url = session.pop("next_url", url_for("home"))
    return redirect(next_url)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/policy')
def policy():
    return render_template("policy.html")

@app.route('/tutorial')
def tutorial():
    return render_template("tutorial.html")

@app.route('/community', methods=['GET'])
@login_required
def community():
    # optional search
    q = request.args.get('q', None)
    if q:
        posts = Post.query.filter(Post.content.contains(q)).order_by(Post.timestamp.desc()).all()
    else:
        posts = Post.query.order_by(Post.timestamp.desc()).all()

    # Attach user info from the 'users' dict
    for p in posts:
        # p.user_id is the Google ID
        # Check if we have that ID in memory
        author = users.get(p.user_id, None)
        if author:
            # If found, add two “fake” attributes to the post object
            # so the template can display them
            p.display_name = author.name
            p.profile_pic = author.profile_pic
        else:
            # fallback to default if user not found
            p.display_name = "Unknown"
            p.profile_pic = url_for('static', filename='images/avatar.png')

    return render_template('community.html', posts=posts)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    try:
        # Build the user's content
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=user_input)],
            )
        ]

        # Create the config object with all the parameters you want:
        generation_config = types.GenerateContentConfig(
            system_instruction=(
                "You are NeoTrade AI, a friendly trading assistant. "
                "Speak in a warm, natural tone using short paragraphs "
                "Keep sentences easy to read."
                "Do in paragraph form"
                "Answer in simple answer"
            ),
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=1024,
            response_mime_type="text/plain"
        )

        # We'll make an async call with an event loop
        loop = asyncio.get_event_loop()
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=generation_config
        )

        return jsonify({"reply": response.text.strip()})

    except Exception as e:
        print("Gemini API Error:", e)
        return jsonify({"reply": "Sorry, there was an error generating a response."})

@app.route('/tchat', methods=['POST'])
def tchat():
    user_input = request.json.get("message", "")
    try:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=user_input)]
            )
        ]

        generation_config = types.GenerateContentConfig(
            system_instruction=(
                "You are Tchat, a friendly tutorial assistant..."
                "Answer in paragrapg form"
            ),
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=1024
        )

        # Just call generate_content directly, no async needed:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=generation_config
        )

        return jsonify({"reply": response.text.strip()})
    except Exception as e:
        print("Gemini API Error in Tchat route:", e)
        return jsonify({"reply": "Sorry, there was an error generating a response."})

# =====================
# Additional Community Routes (Feed, New Post, and New Comment)
# =====================
@app.route('/community/feed', methods=['GET'])
@login_required
def community_feed():
    # optional search via ?q=...
    q = request.args.get('q', None)
    if q:
        posts = Post.query.filter(Post.content.contains(q)).order_by(Post.timestamp.desc()).all()
    else:
        posts = Post.query.order_by(Post.timestamp.desc()).all()
    # Render a template named 'community.html' that you'll need to create
    return render_template('community.html', posts=posts)

@app.route('/community/new_post', methods=['POST'])
@login_required
def new_post():
    content = request.form.get('content', '').strip()
    if content:
        newpost = Post(
            user_id=current_user.id,
            content=content
        )
        db.session.add(newpost)
        db.session.commit()
    # After creating a post, return to the feed
    return redirect(url_for('community_feed'))

@app.route('/community/<int:post_id>/comment', methods=['POST'])
@login_required
def add_comment(post_id):
    post = Post.query.get_or_404(post_id)
    comment_text = request.form.get('comment_content', '').strip()
    if comment_text:
        new_comment = Comment(
            user_id=current_user.id,
            post_id=post_id,
            content=comment_text
        )
        db.session.add(new_comment)
        db.session.commit()
    return redirect(url_for('community_feed'))


@app.route('/trading')
@login_required
def trading():
    try:
        # Get live current price from Binance API
        response = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": "BTCUSDT"})
        data = response.json()
        current_price = round(float(data["price"]), 2)
    except Exception as e:
        print("Error fetching price from Binance:", e)
        current_price = "-"

    signal_data = {
        "current_price": current_price,
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signal": "HOLD",
        "entry_price": "-",
        "stop_loss": "-",
        "take_profit": "-",
        "pnl": "-"
    }

    return render_template("trading.html", signal=signal_data)

@app.route('/predict', methods=['POST'])
def predict():
    global rolling_data

    data = request.json
    price = float(data['price'])

    # Simulate a live candle row — You may want to replace with real OHLCV websocket data
    new_row = pd.DataFrame([{
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "volume": 1.0  # fake volume just to let feature creation work
    }])

    # Append to rolling window
    rolling_data = pd.concat([rolling_data, new_row], ignore_index=True).tail(295)

    if len(rolling_data) < 24:
        return jsonify({
            "signal": "HOLD",
            "stop_loss": price * 0.99,
            "take_profit": price * 1.01
        })

    # Create features using the latest rolling_data
    feature_data = create_features(rolling_data.copy())
    feature_data = feature_data.tail(24)

    if feature_data.empty or feature_data.shape[0] < 24:
        return jsonify({
            "signal": "HOLD",
            "stop_loss": price * 0.99,
            "take_profit": price * 1.01
        })

    # Step 1: CNN feature extraction
    X_cnn = feature_data.values.reshape(1, 24, feature_data.shape[1])
    cnn_features = cnn_model.predict(X_cnn).reshape(1, -1)

    # Step 2: Add HMM regime detection
    regime = hmm_model.predict(cnn_features).reshape(-1, 1)
    hybrid_input = np.concatenate([cnn_features, regime], axis=1)

    # Step 3: Predict signal
    probs = hybrid_model.predict(hybrid_input)[0]  # shape: (3,)
    labels = ["BUY", "SELL", "HOLD"]
    predicted_index = np.argmax(probs)
    signal = labels[predicted_index]

    # Step 4: Calculate SL/TP
    sl = price * (1 - 0.01) if signal == "BUY" else price * (1 + 0.01)
    tp = price * (1 + 0.02) if signal == "BUY" else price * (1 - 0.02)

    return jsonify({
        "signal": signal,
        "stop_loss": round(sl, 2),
        "take_profit": round(tp, 2)
    })

@app.route("/latest_signal", methods=["GET"])
def latest_signal():
    if signals.empty or data_current_df.empty:
        return jsonify({
            "signal": "HOLD",
            "entry_price": "-",
            "stop_loss": "-",
            "take_profit": "-",
            "pnl": "-",
            "current_price": round(data_current_df['close'].iloc[-1], 2) if not data_current_df.empty else "-",
            "last_updated": last_model_update or "-"
        })

    signal_row = signals.iloc[-1]
    current_price = data_current_df['close'].iloc[-1]
    entry = signal_row['entry_price']
    pnl = ((current_price - entry) / entry * 100) if signal_row['trade_type'] == 'BUY' else ((entry - current_price) / entry * 100)

    return jsonify({
        "signal": signal_row['trade_type'],
        "entry_price": round(entry, 2),
        "stop_loss": round(signal_row['stop_loss'], 2),
        "take_profit": round(signal_row['take_profit'], 2),
        "current_price": round(current_price, 2),
        "pnl": round(pnl, 2),
        "last_updated": last_model_update or "-"
    })

@app.route("/latest_price")
def latest_price():
    try:
        response = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT")
        data = response.json()
        price = round(float(data["price"]), 2)
        return jsonify({"price": price})
    except Exception as e:
        print("Error fetching price:", e)
        return jsonify({"price": "-"})


@app.route("/signal_last_updated")
def signal_last_updated():
    return jsonify({"last_updated": last_model_update})


if __name__ == '__main__':
    print("Running app...")
    socketio.run(app, debug=True)
