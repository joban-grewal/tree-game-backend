import os
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import google.generativeai as genai
import tflite_runtime.interpreter as tflite
from PIL import Image # Pillow is now used directly
import numpy as np
from dotenv import load_dotenv

# --- 1. SETUP AND CONFIGURATION ---
load_dotenv() # Load environment variables from a .env file

app = Flask(__name__)
CORS(app) # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- NEW: Health Check Flag ---
# This variable will track if our heavy model is ready.
MODEL_IS_LOADED = False

# --- 2. CONFIGURE AI MODELS (SECURELY) ---
# FIXED: Load API keys safely from environment variables
GEMINI_API_KEY = os.environ.get('AIzaSyAfEBbRrXlxwqgbvOXM9SblWJukYQc7sBc')
PLANT_ID_API_KEY = os.environ.get('1VkeVK7iJfw6AbMXkdxZoePg9Ys55eFWMQubuKRMdivwtyzTKD')

if not GEMINI_API_KEY or not PLANT_ID_API_KEY:
    raise ValueError("Crucial API key is missing! Please set GEMINI_API_KEY and PLANT_ID_API_KEY in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# Load your trained health model
# --- CHANGED: Load the TFLite model ---
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
class_names = [
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]
print("Health model loaded successfully!")
# --- NEW: Flip the flag to True now that the model is loaded ---
MODEL_IS_LOADED = True

# In-memory storage for demo purposes
user_points_storage = {}

# --- 3. HELPER FUNCTIONS ---

# --- REWRITTEN: predict_health function for TFLite ---
def predict_health(image_path):
    try:
        # Load and preprocess the image
        img = Image.open(image_path).resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Set the tensor, invoke the interpreter, and get results
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * np.max(predictions[0]), 2)
        
        disease_name = predicted_class.split('___')[-1].replace('_', ' ')
        plant_name = predicted_class.split('___')[0].replace('_', ' ')
        diagnosis = f"{plant_name}: {disease_name} (Confidence: {confidence}%)"
        
        return diagnosis, disease_name, plant_name
    except Exception as e:
        print(f"Error in TFLite health prediction: {e}")
        return "Health status could not be determined.", None, None

def get_care_advice(disease_name, plant_name):
    # Your get_care_advice function is good, no changes needed here.
    if "healthy" in disease_name.lower():
        return "The plant appears healthy! Keep up the great work. Ensure it gets regular watering, adequate sunlight, and balanced nutrients to stay strong."
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        You are a helpful assistant for farmers in Punjab, India. A plant has been diagnosed.
        Plant Type: {plant_name}, Disease/Issue: {disease_name}.
        Provide a concise, actionable care plan with sections for Immediate Actions, Organic Treatment, Nutrient Support, and Prevention. Format the response using simple HTML.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating care advice with Gemini: {e}")
        return "Could not generate care advice at this time."

def get_wikipedia_summary(title):
    # Your get_wikipedia_summary is good, no changes needed.
    headers = {'User-Agent': 'TreeGuardianGame/1.0'}
    base_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    try:
        url = base_url + title.replace(' ', '_')
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get('extract', 'No Wikipedia summary available.')
    except Exception as e:
        print(f"Error fetching Wikipedia summary: {e}")
    return 'No Wikipedia summary available.'

# --- 4. API ENDPOINTS ---


# --- NEW: Smart Health Check Endpoint ---
@app.route('/healthz')
def health_check():
    # This endpoint will be pinged by Render.
    # It will fail until the model is loaded, then it will succeed.
    if MODEL_IS_LOADED:
        return jsonify({"status": "ok"}), 200
    else:
        # Return a 503 Service Unavailable status while the model is loading
        return jsonify({"status": "model_loading"}), 503

@app.route('/upload', methods=['POST'])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    name = secure_filename(image_file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    image_file.save(path)

    with open(path, "rb") as img_f:
        img_b64 = base64.b64encode(img_f.read()).decode("ascii")

    url = "https://api.plant.id/v2/identify"
    response = requests.post(
        url,
        headers={"Content-Type": "application/json", "Api-Key": PLANT_ID_API_KEY}, # FIXED: Use env variable
        json={"images": [img_b64]},
        timeout=45,
    )
    info = response.json()
    
    species = info['suggestions'][0]['plant_name'] if info.get('suggestions') else "Unknown"
    confidence = int(info['suggestions'][0]['probability'] * 100) if info.get('suggestions') else 0
    wiki_summary = get_wikipedia_summary(species)

    user_points = user_points_storage.get('Demo User', 0) + 1
    user_points_storage['Demo User'] = user_points

    return jsonify({
        "success": True, "filename": name,
        "info": {"species": species, "confidence": confidence, "wiki_summary": wiki_summary},
        "points": user_points
    })

@app.route('/diagnose', methods=['POST'])
def diagnose():
    if "image" not in request.files: return jsonify({"success": False, "error": "No image uploaded"}), 400
    
    image_file = request.files["image"]
    name = secure_filename(image_file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    image_file.save(path)

    diagnosis, disease_name, plant_name = predict_health(path)
    if not disease_name: return jsonify({"success": False, "error": "Prediction failed"})
    
    care_advice = get_care_advice(disease_name, plant_name)

    return jsonify({"success": True, "diagnosis": diagnosis, "care_advice": care_advice})

# ... (Your leaderboard endpoint can remain here) ...
@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    # A simplified leaderboard using the in-memory storage
    demo_points = user_points_storage.get('Demo User', 0)
    
    # You can add some dummy users to make it look more populated
    leaderboard_data = [
        {"user": "Bot Alpha", "points": 15},
        {"user": "Demo User", "points": demo_points},
        {"user": "Bot Beta", "points": 8},
    ]
    
    # Sort the list by points in descending order
    leaderboard_data.sort(key=lambda x: x['points'], reverse=True)
    
    return jsonify(leaderboard_data)

# --- 5. RUN THE APP ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)