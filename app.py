from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import requests
import base64

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = '../uploads'

@app.route('/')
def index():
    return jsonify({"message": "API is working!"})

@app.route('/upload', methods=['POST'])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    image = request.files["image"]
    name = secure_filename(image.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    image.save(path)

    # Call Plant.id API (existing code)
    with open(path, "rb") as img_f:
        img_b64 = base64.b64encode(img_f.read()).decode("ascii")
    api_key = "<YOUR_PLANTID_API_KEY>"  # replace
    url = "https://api.plant.id/v2/identify"
    response = requests.post(
        url,
        headers={"Content-Type": "application/json", "Api-Key": api_key},
        json={
            "images": [img_b64],
            "organs": ["leaf"],
            "details": ["common_names", "url", "name_authority", "wiki_description", "taxonomy"]
        },
        timeout=45,
    )
    info = response.json()
    if info.get('suggestions'):
        species = info['suggestions'][0].get('plant_name', 'Unknown')
        confidence = int(info['suggestions'][0].get('probability', 0) * 100)
        facts = info['suggestions'][0].get('plant_details', {}).get('wiki_description', {}).get('value', 'No facts found.')
    else:
        species = "Unknown"
        confidence = 0
        facts = "No facts found."

    # --- Point system logic ---
    user_id = request.form.get('user_id', 'demo_user')  # For demo, use a static user
    user_points = get_user_points(user_id)  # Function to get current points
    user_points += 1  # Each tree found = 1 point
    set_user_points(user_id, user_points)  # Save back to storage

    return jsonify({
        "success": True,
        "filename": name,
        "info": {
            "species": species,
            "confidence": confidence,
            "facts": facts
        },
        "points": user_points
    })

# Helper functions for points (demo, in-memory storage)
user_points_storage = {}  # For demo, replace with actual persistent DB

def get_user_points(user_id):
    return user_points_storage.get(user_id, 0)

def set_user_points(user_id, points):
    user_points_storage[user_id] = points

@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    # simulate leaderboard with static or stored user points
    leaderboard_data = [
        {"user": "User1", "points": get_user_points("user1")},
        {"user": "User2", "points": get_user_points("user2")},
        {"user": "Demo User", "points": get_user_points("demo_user")},
    ]
    # Sort by points desc
    leaderboard_data.sort(key=lambda x: x['points'], reverse=True)
    return jsonify(leaderboard_data)


if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
