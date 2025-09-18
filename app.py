from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import requests
import base64


def get_wikipedia_summary(title):
    """Fetch Wikipedia summary for tree species (common/scientific name)."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('extract', 'No Wikipedia summary available.')
    return 'No Wikipedia summary available.'


app = Flask(__name__)
# Enable CORS for all origins (good for testing)
CORS(app, resources={r"/*": {"origins": "*"}})

app.config['UPLOAD_FOLDER'] = '../uploads'

# In-memory points storage for demo (replace with real DB later)
user_points_storage = {}


def get_user_points(user_id):
    return user_points_storage.get(user_id, 0)


def set_user_points(user_id, points):
    user_points_storage[user_id] = points


@app.route('/')
def index():
    return jsonify({"message": "API is working!"})


@app.route('/upload', methods=['POST'])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    name = secure_filename(image.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    image.save(path)

    with open(path, "rb") as img_f:
        img_b64 = base64.b64encode(img_f.read()).decode("ascii")

    api_key = "1VkeVK7iJfw6AbMXkdxZoePg9Ys55eFWMQubuKRMdivwtyzTKD"
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

    # Fetch Wikipedia summary for the tree species
    wiki_summary = get_wikipedia_summary(species)

    # Points system - static user for demo
    user_id = request.form.get('user_id', 'demo_user')
    user_points = get_user_points(user_id)
    user_points += 1
    set_user_points(user_id, user_points)

    return jsonify({
        "success": True,
        "filename": name,
        "info": {
            "species": species,
            "confidence": confidence,
            "facts": facts,
            "wiki_summary": wiki_summary
        },
        "points": user_points
    })


@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    leaderboard_data = [
        {"user": "User1", "points": get_user_points("user1")},
        {"user": "User2", "points": get_user_points("user2")},
        {"user": "Demo User", "points": get_user_points("demo_user")},
    ]
    leaderboard_data.sort(key=lambda x: x['points'], reverse=True)
    return jsonify(leaderboard_data)


if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
