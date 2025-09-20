import os
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import google.generativeai as genai
import tensorflow as tf
from PIL import Image
import numpy as np
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import json
import hashlib
import time
from functools import wraps
import re

# ===== CONFIGURATION =====
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# App configuration
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# API Keys
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
PLANT_ID_API_KEY = os.environ.get('PLANT_ID_API_KEY')

if not GEMINI_API_KEY or not PLANT_ID_API_KEY:
    logger.error("Critical: API keys missing!")
    raise ValueError("GEMINI_API_KEY and PLANT_ID_API_KEY must be set")

genai.configure(api_key=GEMINI_API_KEY)

# ===== MODEL LOADING =====
MODEL_IS_LOADED = False

try:
    logger.info("Loading TensorFlow Lite model...")
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    class_names = [
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]
    
    MODEL_IS_LOADED = True
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    MODEL_IS_LOADED = False

# ===== IN-MEMORY STORAGE =====
class GameDatabase:
    def __init__(self):
        self.users = {}
        self.leaderboard = []
        self.achievements = {}
        self.daily_challenges = {}
        self.rate_limits = {}
        
    def get_user(self, user_id):
        if user_id not in self.users:
            self.users[user_id] = {
                'id': user_id,
                'username': f'User_{user_id[:8]}',
                'points': 0,
                'level': 1,
                'experience': 0,
                'trees_identified': 0,
                'achievements': [],
                'streak': 0,
                'last_active': datetime.now().isoformat(),
                'collection': []
            }
        return self.users[user_id]
    
    def update_user(self, user_id, data):
        user = self.get_user(user_id)
        user.update(data)
        self.users[user_id] = user
        self.update_leaderboard()
    
    def update_leaderboard(self):
        self.leaderboard = sorted(
            [{'user': u['username'], 'points': u['points'], 'level': u['level']} 
             for u in self.users.values()],
            key=lambda x: x['points'],
            reverse=True
        )[:100]  # Top 100
    
    def add_achievement(self, user_id, achievement_id):
        user = self.get_user(user_id)
        if achievement_id not in user['achievements']:
            user['achievements'].append(achievement_id)
            user['points'] += 50
            self.update_user(user_id, user)

db = GameDatabase()

# ===== HELPER FUNCTIONS =====
def generate_user_id(request):
    """Generate unique user ID from request"""
    ip = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')
    unique_string = f"{ip}{user_agent}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def rate_limit(max_requests=10, window=60):
    """Simple rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            user_id = generate_user_id(request)
            now = time.time()
            
            if user_id not in db.rate_limits:
                db.rate_limits[user_id] = []
            
            # Clean old requests
            db.rate_limits[user_id] = [
                t for t in db.rate_limits[user_id]
                if now - t < window
            ]
            
            if len(db.rate_limits[user_id]) >= max_requests:
                return jsonify({
                    'success': False,
                    'error': 'Rate limit exceeded. Please wait.'
                }), 429
            
            db.rate_limits[user_id].append(now)
            return f(*args, **kwargs)
        return wrapped
    return decorator

def calculate_points(confidence, is_rare=False):
    """Calculate points based on confidence and rarity"""
    base_points = 10
    confidence_bonus = int(confidence / 10)
    rarity_bonus = 20 if is_rare else 0
    return base_points + confidence_bonus + rarity_bonus

def normalize_plant_name(name):
    """Normalize plant names for matching"""
    if not name:
        return ""
    name = re.sub(r'[^\w\s]', '', name.lower())
    replacements = {
        'corn': 'maize',
        'maize': 'corn',
        'tomatoes': 'tomato',
        'tomato plant': 'tomato',
        'potatoes': 'potato',
        'potato plant': 'potato'
    }
    for variant, normalized in replacements.items():
        if variant in name:
            return normalized
    
    # Check for partial matches
    for key in ['corn', 'tomato', 'potato']:
        if key in name:
            return key
    
    return name

def predict_health(image_path, expected_plant):
    """Enhanced health prediction with better accuracy"""
    if not MODEL_IS_LOADED:
        return "Health service temporarily unavailable", None, None
    
    try:
        # CRITICAL: Proper image preprocessing to match training data
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            if img.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            else:
                img = img.convert('RGB')
        
        # IMPORTANT: Resize to exact model input size
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to array and normalize EXACTLY as training data
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Enhanced plant matching with multiple strategies
        normalized_expected = normalize_plant_name(expected_plant)
        
        # Strategy 1: Direct matching
        matching_indices = []
        for i, class_name in enumerate(class_names):
            class_parts = class_name.lower().split('___')[0].replace('_', ' ')
            
            # Check for exact plant type match
            if 'corn' in normalized_expected or 'maize' in normalized_expected:
                if 'corn' in class_parts or 'maize' in class_parts:
                    matching_indices.append(i)
            elif 'tomato' in normalized_expected:
                if 'tomato' in class_parts:
                    matching_indices.append(i)
            elif 'potato' in normalized_expected:
                if 'potato' in class_parts:
                    matching_indices.append(i)
        
        # Strategy 2: If no matches, try broader search
        if not matching_indices:
            # Check if ANY of our supported plants match
            for plant in ['corn', 'maize', 'tomato', 'potato']:
                if plant in normalized_expected:
                    for i, class_name in enumerate(class_names):
                        if plant in class_name.lower():
                            matching_indices.append(i)
                    break
        
        # Strategy 3: If still no matches, use all predictions for closest plant
        if not matching_indices:
            # Find the most likely plant type from all predictions
            top_prediction_idx = np.argmax(predictions)
            predicted_plant = class_names[top_prediction_idx].split('___')[0].replace('_', ' ')
            
            logger.warning(f"No exact match for '{expected_plant}', using top prediction: {predicted_plant}")
            
            # Get all indices for the top predicted plant type
            for i, class_name in enumerate(class_names):
                if predicted_plant.lower() in class_name.lower():
                    matching_indices.append(i)
        
        if not matching_indices:
            return f"Unable to diagnose. Please ensure the image shows a tomato, potato, or corn plant.", None, None
        
        # Get the best prediction from matching classes
        matching_predictions = predictions[matching_indices]
        best_match_index = np.argmax(matching_predictions)
        final_class_index = matching_indices[best_match_index]
        
        predicted_class = class_names[final_class_index]
        confidence = round(float(matching_predictions[best_match_index]) * 100, 2)
        
        # Parse the prediction
        parts = predicted_class.split('___')
        plant_name = parts[0].replace('_', ' ').replace('(', '').replace(')', '')
        
        if len(parts) > 1:
            disease_name = parts[1].replace('_', ' ')
            # Clean up disease name
            disease_name = disease_name.replace('  ', ' ').strip()
        else:
            disease_name = 'Unknown condition'
        
        # Create detailed diagnosis
        if confidence < 30:
            diagnosis = f"Low confidence ({confidence}%) - Results may not be accurate. Please use a clearer image of the plant leaves."
        elif 'healthy' in disease_name.lower():
            diagnosis = f"✅ {plant_name} appears healthy! (Confidence: {confidence}%)"
        else:
            diagnosis = f"⚠️ {plant_name} - Detected: {disease_name} (Confidence: {confidence}%)"
        
        logger.info(f"Health prediction: {diagnosis}")
        
        return diagnosis, disease_name, plant_name
        
    except Exception as e:
        logger.error(f"Health prediction error: {e}", exc_info=True)
        return "Unable to analyze image. Please ensure the image clearly shows plant leaves.", None, None

def get_care_advice(disease_name, plant_name):
    """Generate care advice using Gemini"""
    if "healthy" in disease_name.lower():
        return """
        <div class='care-success'>
            <h4>✅ Your plant is healthy!</h4>
            <ul>
                <li>🌱 Maintain regular watering schedule</li>
                <li>☀️ Ensure adequate sunlight</li>
                <li>🌿 Monitor for early signs of stress</li>
                <li>💚 Apply balanced fertilizer monthly</li>
            </ul>
        </div>
        """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Plant: {plant_name}
        Disease: {disease_name}
        
        Provide care advice in this HTML format:
        <div class='care-advice'>
            <h4>Immediate Actions:</h4>
            <ul><li>2-3 urgent steps</li></ul>
            <h4>Treatment:</h4>
            <ul><li>Organic and chemical options</li></ul>
            <h4>Prevention:</h4>
            <ul><li>Future prevention tips</li></ul>
        </div>
        Keep it concise and practical.
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return """
        <div class='care-advice'>
            <h4>General Care Guidelines:</h4>
            <ul>
                <li>🔬 Isolate affected plants</li>
                <li>✂️ Remove diseased parts</li>
                <li>💧 Adjust watering schedule</li>
                <li>🌬️ Improve air circulation</li>
                <li>🧪 Consider organic fungicides</li>
            </ul>
        </div>
        """

def get_wikipedia_summary(title):
    """Fetch Wikipedia summary"""
    try:
        clean_title = title.split('(')[0].strip().replace(' ', '_')
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean_title}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            extract = response.json().get('extract', '')
            return extract[:300] + '...' if len(extract) > 300 else extract
    except Exception as e:
        logger.warning(f"Wikipedia error: {e}")
    
    return f"{title} is an interesting plant species. Keep exploring!"

# ===== API ENDPOINTS =====

@app.route('/healthz')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if MODEL_IS_LOADED else 'degraded',
        'model': 'loaded' if MODEL_IS_LOADED else 'loading',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0'
    }), 200 if MODEL_IS_LOADED else 503

@app.route('/upload', methods=['POST'])
@rate_limit(max_requests=20, window=60)
def upload():
    """Upload and identify plant"""
    try:
        user_id = generate_user_id(request)
        user = db.get_user(user_id)
        
        # Check file
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if not allowed_file(image_file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Save file
        filename = secure_filename(image_file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)
        
        # Optimize image
        with Image.open(filepath) as img:
            if img.size[0] > 1024 or img.size[1] > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                img.save(filepath, optimize=True, quality=85)
        
        # Prepare for Plant.ID
        with open(filepath, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('ascii')
        
        # Call Plant.ID API
        logger.info(f"Identifying plant for user {user_id}")
        response = requests.post(
            'https://api.plant.id/v2/identify',
            headers={'Content-Type': 'application/json', 'Api-Key': PLANT_ID_API_KEY},
            json={'images': [img_b64], 'plant_details': ['common_names']},
            timeout=30
        )
        
        if response.status_code != 200:
            return jsonify({'success': False, 'error': 'Identification service unavailable'}), 503
        
        data = response.json()
        
        # Parse results
        if data.get('suggestions'):
            suggestion = data['suggestions'][0]
            species = suggestion.get('plant_name', 'Unknown')
            confidence = int(suggestion.get('probability', 0) * 100)
            common_names = suggestion.get('plant_details', {}).get('common_names', [])
            
            if common_names:
                species += f" ({', '.join(common_names[:2])})"
        else:
            species = 'Unknown'
            confidence = 0
        
        # Calculate rewards
        points_earned = calculate_points(confidence)
        experience_earned = 20
        
        # Update user stats
        user['points'] += points_earned
        user['experience'] += experience_earned
        user['trees_identified'] += 1
        user['last_active'] = datetime.now().isoformat()
        
        # Check for level up
        if user['experience'] >= user['level'] * 100:
            user['level'] += 1
            user['experience'] = 0
            points_earned += 100  # Level up bonus
        
        # Add to collection
        user['collection'].append({
            'species': species,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
        
        # Check achievements
        if user['trees_identified'] == 1:
            db.add_achievement(user_id, 'first_tree')
        elif user['trees_identified'] == 10:
            db.add_achievement(user_id, 'collector')
        elif user['trees_identified'] == 50:
            db.add_achievement(user_id, 'expert')
        
        db.update_user(user_id, user)
        
        # Get Wikipedia info
        wiki_summary = get_wikipedia_summary(species.split('(')[0].strip())
        
        # Clean up old file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'success': True,
            'filename': filename,
            'info': {
                'species': species,
                'confidence': confidence,
                'wiki_summary': wiki_summary
            },
            'points_earned': points_earned,
            'user_stats': {
                'points': user['points'],
                'level': user['level'],
                'experience': user['experience']
            }
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'success': False, 'error': 'Processing failed'}), 500

@app.route('/diagnose', methods=['POST'])
@rate_limit(max_requests=10, window=60)
def diagnose():
    """Diagnose plant health"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        species = request.form.get('species', 'Unknown')
        image_file = request.files['image']
        
        # Save file
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"health_{filename}")
        image_file.save(filepath)
        
        # Get diagnosis
        diagnosis, disease, plant = predict_health(filepath, species)
        
        if not disease:
            return jsonify({'success': False, 'error': diagnosis})
        
        # Get care advice
        care_advice = get_care_advice(disease, plant)
        
        # Update user stats
        user_id = generate_user_id(request)
        user = db.get_user(user_id)
        user['points'] += 5  # Points for health check
        db.update_user(user_id, user)
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'success': True,
            'diagnosis': diagnosis,
            'care_advice': care_advice,
            'plant_name': plant,
            'condition': disease
        })
        
    except Exception as e:
        logger.error(f"Diagnosis error: {e}")
        return jsonify({'success': False, 'error': 'Diagnosis failed'}), 500

@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    """Get leaderboard data"""
    try:
        # Add some bot players for engagement
        bots = [
            {'user': 'TreeMaster Pro', 'points': 2500, 'level': 15},
            {'user': 'Forest Guardian', 'points': 2000, 'level': 12},
            {'user': 'Leaf Expert', 'points': 1500, 'level': 10},
            {'user': 'Nature Wizard', 'points': 1000, 'level': 8},
            {'user': 'Plant Doctor', 'points': 750, 'level': 6}
        ]
        
        # Combine real users and bots
        all_players = db.leaderboard + bots
        all_players.sort(key=lambda x: x['points'], reverse=True)
        
        return jsonify(all_players[:20])  # Top 20
        
    except Exception as e:
        logger.error(f"Leaderboard error: {e}")
        return jsonify([]), 500

@app.route('/user/stats', methods=['GET'])
def user_stats():
    """Get user statistics"""
    try:
        user_id = generate_user_id(request)
        user = db.get_user(user_id)
        
        return jsonify({
            'success': True,
            'stats': {
                'points': user['points'],
                'level': user['level'],
                'experience': user['experience'],
                'trees_identified': user['trees_identified'],
                'achievements': user['achievements'],
                'streak': user['streak']
            }
        })
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'success': False}), 500

@app.route('/achievements', methods=['GET'])
def achievements():
    """Get achievements list"""
    return jsonify({
        'achievements': [
            {'id': 'first_tree', 'name': 'First Discovery', 'points': 50},
            {'id': 'collector', 'name': 'Tree Collector', 'points': 50},
            {'id': 'expert', 'name': 'Forest Expert', 'points': 100},
            {'id': 'streak_week', 'name': 'Week Warrior', 'points': 75},
            {'id': 'perfect_diagnosis', 'name': 'Plant Doctor', 'points': 50}
        ]
    })

@app.route('/debug/health', methods=['POST'])
def debug_health():
    """Debug endpoint to test health prediction"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    image = request.files['image']
    plant_type = request.form.get('plant', 'tomato')  # Default to tomato
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'debug_health.jpg')
    image.save(filepath)
    
    # Get raw predictions
    img = Image.open(filepath).resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Get top 5 predictions
    top_5 = np.argsort(predictions)[-5:][::-1]
    results = []
    for idx in top_5:
        results.append({
            'class': class_names[idx],
            'confidence': float(predictions[idx])
        })
    
    os.remove(filepath)
    
    return jsonify({
        'plant_type': plant_type,
        'top_predictions': results
    })

# ===== ERROR HANDLERS =====
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large (max 10MB)'}), 413

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

# ===== MAIN =====
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌳 Tree Guardian API starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)