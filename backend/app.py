from flask import Flask, request, render_template, redirect, url_for, session
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import errno
from pymongo import MongoClient
from datetime import datetime
import bcrypt
import shutil
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
# Configure Flask Session (server-side)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
from flask_session import Session
Session(app)

# --- Modified Session Folder Handling ---
# Ensure session folder exists, handle potential permission errors on deletion
SESSION_FOLDER = './.flask_session/'

# Function to attempt safe removal of the session folder
def safe_remove_session_folder(folder_path):
    """Attempts to remove the session folder, handling common errors."""
    if os.path.exists(folder_path):
        try:
            # Try to remove the directory tree
            shutil.rmtree(folder_path)
        except OSError as e:
            # Common error on Windows if folder is in use or permission denied
            if e.errno == errno.EACCES or e.errno == errno.EPERM:
                print(f"Warning: Could not remove session folder '{folder_path}' due to access permissions or it being in use. Ignoring for now.")
                # Optionally, try again after a short delay if needed, or just warn.
                # For now, we warn and proceed, letting Flask handle existing sessions potentially.
            else:
                # Re-raise if it's a different kind of OS error
                raise

# Attempt to clean up the session folder on startup, but don't crash if it fails
safe_remove_session_folder(SESSION_FOLDER)

# Ensure the folder exists for Flask-Session to use
try:
    os.makedirs(SESSION_FOLDER, exist_ok=True)
except OSError as e:
    if e.errno != errno.EEXIST: # If the error isn't just that it already exists
        print(f"Critical Error: Could not create session folder '{SESSION_FOLDER}': {e}")
        raise # Re-raise critical errors

# --- End of Modified Session Handling ---

# Load Model
MODEL_PATH = r"model/crop_disease_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)
# Dynamically fetch class names
dataset_path = r"C:\Users\91808\OneDrive\Desktop\PlantVillage\train"
class_names = sorted(os.listdir(dataset_path))
# Cure Suggestions
cure_suggestions = {
    "Tomato___Bacterial_spot": "Apply copper-based bactericides like Bordeaux mixture. Ensure proper spacing between plants to improve airflow and reduce moisture.",
    "Tomato___Early_blight": "Spray Mancozeb or Chlorothalonil weekly. Remove and destroy infected leaves to prevent spread.",
    "Tomato___Late_blight": "Use metalaxyl-based fungicides like Ridomil Gold. Immediately remove infected plants and avoid watering overhead.",
    "Tomato___Leaf_Mold": "Ensure good air circulation. Apply sulfur-based sprays and remove affected leaves promptly.",
    "Tomato___Septoria_leaf_spot": "Apply fungicides like Chlorothalonil or Mancozeb every 7–10 days. Mulch to prevent soil splash.",
    "Tomato___Spider_mites__Two_spotted_spider_mite": "Spray neem oil or insecticidal soap. Keep humidity high as mites thrive in dry conditions.",
    "Tomato___Target_Spot": "Use strobilurin fungicides like pyraclostrobin. Improve air flow and rotate crops.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Use resistant tomato varieties. Control whiteflies using yellow sticky traps or neem spray.",
    "Tomato___Tomato_mosaic_virus": "Remove and destroy infected plants. Disinfect tools regularly. Avoid smoking near plants.",
    "Tomato___Healthy": "Your tomato plant is healthy! Continue good practices: water at the base, prune, and fertilize moderately.",
    "Potato___Early_blight": "Use Mancozeb or Chlorothalonil. Remove affected leaves. Practice crop rotation to prevent recurrence.",
    "Potato___Late_blight": "Apply fungicides like metalaxyl. Ensure good drainage and avoid excessive watering.",
    "Potato___Healthy": "Your potato plant is in good health. Maintain regular monitoring and avoid over-irrigation.",
    "Pepper__bell___Bacterial_spot": "Use copper-based sprays weekly. Avoid splashing water on leaves and rotate crops annually.",
    "Pepper__bell___Healthy": "Water deeply 1–2 times/week. Needs 6+ hours of direct sunlight.",
    "Grape___Black_rot": "Apply fungicides like Myclobutanil or Captan. Prune vines for air circulation and remove fallen leaves.",
    "Grape___Esca_(Black_Measles)": "Prune infected parts. Avoid wounds during pruning. Remove affected vines if necessary.",
    "Grape___Leaf_blight_(Isariopsis)": "Use copper-based fungicides and improve vineyard air flow.",
    "Grape___Healthy": "Grapevine looks great! Keep an eye out for pests and use mulch to retain moisture.",
    "Corn___Cercospora_leaf_spot__Gray_leaf_spot": "Apply triazole or strobilurin fungicides. Avoid planting corn in the same soil consecutively.",
    "Corn___Common_rust_": "Use triazole-based fungicides like Propiconazole. Choose rust-resistant hybrids.",
    "Corn___Northern_Leaf_Blight": "Spray strobilurin-based fungicides. Remove crop debris and avoid late-season nitrogen application.",
    "Corn___Healthy": "Your corn crop is doing well. Provide consistent watering and nitrogen-rich fertilizers.",
    "Apple___Apple_scab": "Apply sulfur or copper-based fungicides during early leaf growth. Remove infected fruits and leaves.",
    "Apple___Black_rot": "Prune infected branches. Remove mummified fruits and sanitize pruning tools.",
    "Apple___Cedar_apple_rust": "Use Myclobutanil fungicide before bud break. Remove nearby junipers if possible.",
    "Apple___Healthy": "Apple tree is thriving. Prune annually and provide 6–8 hours of sun.",
    "Peach___Bacterial_spot": "Spray copper-based fungicides during dormant season. Avoid working in wet foliage.",
    "Peach___Healthy": "Needs 6–8 hours of sun. Water deeply once a week.",
    "Cherry_(including_sour)___Powdery_mildew": "Use sulfur or potassium bicarbonate sprays. Ensure good airflow by pruning.",
    "Cherry_(including_sour)___Healthy": "Full sun, well-drained soil are essential.",
    "Strawberry___Leaf_scorch": "Use fungicides like Chlorothalonil or Captan. Avoid crowded planting.",
    "Strawberry___Healthy": "Keep soil evenly moist. Needs 6–8 hours sunlight daily.",
    "Blueberry___Healthy": "Healthy blueberry plant! Maintain acidic soil (pH 4.5–5.5) and avoid waterlogging.",
    "Raspberry___Healthy": "Your raspberry plant is thriving. Ensure trellising and regular pruning for best fruit yield.",
    "Soybean___Healthy": "Soybeans are healthy. Practice crop rotation and monitor regularly for aphids or fungal issues.",
    "Squash___Powdery_mildew": "Apply sulfur-based fungicides. Water early in the day and avoid overcrowding."
}
# Growth Tips for Healthy Crops
growth_tips = {
    "Tomato___Healthy": "Water 1–2 inches per week. Needs 6–8 hours of direct sunlight.",
    "Potato___Healthy": "Keep soil evenly moist. Requires full sun and cool temperatures.",
    "Pepper__bell___Healthy": "Water deeply 1–2 times/week. Needs 6+ hours of sunlight.",
    "Grape___Healthy": "Provide full sun exposure. Use proper trellis support for growth.",
    "Corn___Healthy": "Needs 1–1.5 inches of water weekly. Provide full sun and proper spacing.",
    "Apple___Healthy": "Prune annually. Requires 6–8 hours of full sun per day.",
    "Peach___Healthy": "Full sun (6–8 hrs/day). Deep weekly watering preferred.",
    "Cherry_(including_sour)___Healthy": "Full sunlight and well-drained soil are essential.",
    "Strawberry___Healthy": "Keep soil evenly moist. Needs 6–8 hours of sunlight daily."
}
# Care Tips for Diseased Crops
care_tips = {
    # Tomato
    "Tomato___Bacterial_spot": "Avoid waterlogging. Water at base.",
    "Tomato___Early_blight": "Avoid overhead watering.",
    "Tomato___Late_blight": "Ensure good drainage.",
    "Tomato___Leaf_Mold": "Reduce humidity. Improve air circulation.",
    "Tomato___Septoria_leaf_spot": "Keep foliage dry.",
    "Tomato___Spider_mites__Two_spotted_spider_mite": "Increase humidity.",
    "Tomato___Target_Spot": "Avoid wet foliage.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies.",
    "Tomato___Tomato_mosaic_virus": "Disinfect tools.",
    # Potato
    "Potato___Early_blight": "Avoid overwatering.",
    "Potato___Late_blight": "Avoid planting in wet soil.",
    # Pepper
    "Pepper__bell___Bacterial_spot": "Avoid water on leaves. Use clean seeds.",
    # Grape
    "Grape___Black_rot": "Prune to allow airflow.",
    "Grape___Esca_(Black_Measles)": "Improve vineyard drainage.",
    "Grape___Leaf_blight_(Isariopsis)": "Avoid wet foliage.",
    # Corn
    "Corn___Cercospora_leaf_spot__Gray_leaf_spot": "Use clean seed and rotate crops.",
    "Corn___Common_rust_": "Use resistant hybrids.",
    "Corn___Northern_Leaf_Blight": "Ensure field sanitation.",
    # Apple
    "Apple___Apple_scab": "Prune to increase air circulation.",
    "Apple___Black_rot": "Maintain good airflow. Remove fallen debris.",
    "Apple___Cedar_apple_rust": "Avoid high humidity areas. Prune regularly.",
    # Peach
    "Peach___Bacterial_spot": "Use copper sprays during dormancy.",
    # Cherry
    "Cherry_(including_sour)___Powdery_mildew": "Prune regularly.",
    # Strawberry
    "Strawberry___Leaf_scorch": "Avoid overhead watering."
}
# MongoDB Setup
connection_string = "mongodb+srv://sagar:cropdiseasedetection@cropcluster.ityvjxo.mongodb.net/agri_scan?retryWrites=true&w=majority&appName=cropcluster"
client = MongoClient(connection_string)
db = client["agri_scan"]
history_collection = db["prediction_history"]
users_collection = db["users"]
# Preprocessing Function
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)
# Helper Functions
def is_logged_in():
    return 'email' in session
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
def check_password(hashed, password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)
# Routes
@app.route('/')
def index():
    if not is_logged_in():
        return redirect(url_for('signin'))
    try:
        user_history = list(history_collection.find({"email": session['email']}).sort("timestamp", -1).limit(10))
    except Exception as e:
        print("MongoDB Connection Error:", e)
        user_history = []
    return render_template('index.html', prediction=None, history=user_history, session_email=session.get('email'))
@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = users_collection.find_one({"email": email})
        if user and check_password(user['password'], password):
            session.clear()
            session['email'] = email
            session.permanent = False  # Session ends when browser closes
            return redirect(url_for('index'))
        return render_template('signin.html', error="Invalid email or password.")
    return render_template('signin.html', error=None)
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            return render_template('signup.html', error="Email already exists.")
        hashed = hash_password(password)
        users_collection.insert_one({
            "email": email,
            "password": hashed,
            "created_at": datetime.now()
        })
        return redirect(url_for('signin'))
    return render_template('signup.html', error=None)
@app.route('/signout')
def signout():
    session.clear()
    return redirect(url_for('signin'))
@app.route('/predict', methods=['POST'])
def predict():
    if not is_logged_in():
        return redirect(url_for('signin'))
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file selected!")
    file = request.files['file']
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return render_template('index.html', prediction="Unsupported file format!")
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    image = preprocess_image(image)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    confidence = float(np.max(predictions))
    label = class_names[predicted_class]
    result = f"{label} (Confidence: {confidence * 100:.2f}%)"
    # Always show a remedy and tip
    remedy = cure_suggestions.get(label, "Apply general fungicides and monitor regularly. Consult local agricultural expert.")
    if label in growth_tips:
        tip = growth_tips[label]
    elif label in care_tips:
        tip = care_tips[label]
    else:
        tip = "Maintain good crop hygiene. Monitor regularly for pests and diseases."
    # Save prediction to MongoDB
    prediction_data = {
        "email": session['email'],
        "prediction": result,
        "remedy": remedy,
        "tip": tip,
        "timestamp": datetime.now()
    }
    try:
        history_collection.insert_one(prediction_data)
    except Exception as e:
        print("Error saving prediction to MongoDB:", e)
    # Get updated history
    try:
        user_history = list(history_collection.find({"email": session['email']}).sort("timestamp", -1).limit(10))
    except Exception as e:
        print("Error fetching history:", e)
        user_history = []
    return render_template('index.html', prediction=result, remedy=remedy, tip=tip, history=user_history, session_email=session.get('email'))
@app.route('/clear_history')
def clear_history():
    if not is_logged_in():
        return redirect(url_for('signin'))
    history_collection.delete_many({"email": session['email']})
    return index()
# ✅ New About Route
@app.route('/about')
def about():
    return render_template('about.html')
# Run the app
if __name__ == '__main__':
    app.run(debug=True)
