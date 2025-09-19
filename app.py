from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import requests
import base64

import google.generativeai as genai
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# --- 1. CONFIGURE YOUR AI MODELS ---
# Get your Gemini API key from an environment variable for safety
GEMINI_API_KEY = os.environ.get('AIzaSyAfEBbRrXlxwqgbvOXM9SblWJukYQc7sBc')
genai.configure(api_key=GEMINI_API_KEY)

# Load your trained health model
health_model = load_model('punjab_crops_model.h5')

class_names = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# --- 2. DEFINE AI HELPER FUNCTIONS ---
def predict_health(image_path):
    """Loads an image and predicts its health status using your trained model."""
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = health_model.predict(img_array)
        
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * np.max(predictions[0]), 2)
        
        # Clean up the name for better display and for the Gemini prompt
        disease_name = predicted_class.split('___')[-1].replace('_', ' ')
        plant_name = predicted_class.split('___')[0].replace('_', ' ')

        diagnosis = f"{plant_name}: {disease_name} (Confidence: {confidence}%)"
        
        return diagnosis, disease_name, plant_name
    except Exception as e:
        print(f"Error in health prediction: {e}")
        return "Health status could not be determined.", None, None

def get_care_advice(disease_name, plant_name):
    """Generates care advice using the Gemini AI model."""
    if "healthy" in disease_name.lower():
        return "The plant appears healthy! Keep up the great work. Ensure it gets regular watering, adequate sunlight, and balanced nutrients to stay strong."

    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        You are a helpful assistant for farmers and gardeners in Punjab, India.
        A plant has been diagnosed with a health issue.
        
        Plant Type: {plant_name}
        Disease/Issue: {disease_name}

        Provide a concise and actionable care plan. The advice should be practical for a local Punjabi context.
        Include sections for:
        1.  **Immediate Actions:** (e.g., pruning, isolation)
        2.  **Organic Treatment:** (e.g., Neem oil, specific mixtures)
        3.  **Nutrient Support:** (e.g., what nutrients might be lacking or needed)
        4.  **Prevention:** (e.g., watering practices, crop rotation)

        Format the response using simple HTML like <p> and <ul> for lists. Be encouraging.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating care advice with Gemini: {e}")
        return "Could not generate care advice at this time."

# --- 3. SETUP FLASK APP ---
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def get_wikipedia_summary(title):
    """Fetch Wikipedia summary with improved error handling and headers."""
    # CHANGED: Added a User-Agent header to mimic a browser request
    headers = {
        'User-Agent': 'TreeGuardianGame/1.0 (https://yourapp.com; your-email@example.com)'
    }
    
    base_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    search_url = "https://en.wikipedia.org/w/api.php"

    def fetch_summary(page_title):
        try:
            url = base_url + page_title.replace(' ', '_')
            response = requests.get(url, headers=headers, timeout=10) # Added headers
            if response.status_code == 200:
                data = response.json()
                if 'extract' in data:
                    return data['extract']
        except Exception as e:
            # CHANGED: Added print(e) to log the actual error
            print(f"Error fetching direct summary for {page_title}: {e}")
        return None

    # Try direct summary fetch
    summary = fetch_summary(title)
    if summary:
        return summary

    # If direct fetch failed, try search API to find closest page
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': title,
        'format': 'json',
        'srlimit': 1,
    }
    try:
        res = requests.get(search_url, params=params, headers=headers, timeout=10) # Added headers
        data = res.json()
        search_results = data.get('query', {}).get('search', [])
        if search_results:
            best_match_title = search_results[0]['title']
            return fetch_summary(best_match_title)
    except Exception as e:
        # CHANGED: Added print(e) to log the actual error
        print(f"Error during Wikipedia search for {title}: {e}")

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
        # Removed facts as per your request
    else:
        species = "Unknown"
        confidence = 0

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
            "wiki_summary": wiki_summary
        },
        "points": user_points
    })

# --- NEW: DIAGNOSE ENDPOINT ---
@app.route('/diagnose', methods=['POST'])
def diagnose():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    image_file = request.files["image"]
    name = secure_filename(image_file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    image_file.save(path)

    # Step 1: Get diagnosis from your custom model
    diagnosis, disease_name, plant_name = predict_health(path)
    
    if not disease_name:
        return jsonify({"success": False, "error": "Prediction failed"})

    # Step 2: Get care advice from Gemini AI
    care_advice = get_care_advice(disease_name, plant_name)

    return jsonify({
        "success": True,
        "diagnosis": diagnosis,
        "care_advice": care_advice
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
    # Ensure the uploads folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # IMPORTANT: Check if the Gemini API key is set before starting
    # This prevents the app from crashing if the key is missing.
    if not GEMINI_API_KEY:
        raise ValueError("A major issue has been found: GEMINI_API_KEY environment variable not set. Please set it before running the application.")

    # Define the port and run the app
    port = int(os.environ.get("PORT", 5000))
    # Using debug=True is great for local development
    app.run(host="0.0.0.0", port=port, debug=True)