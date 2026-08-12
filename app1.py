import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import json
import os
import sqlite3
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Auto-Chloro Web App",
    page_icon="🌱",
    layout="wide"
)

# Initialize SQLite database
DB_PATH = 'history.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            prediction TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_scan(filename, prediction, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO scan_history (timestamp, filename, prediction, confidence)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, filename, prediction, confidence))
    conn.commit()
    conn.close()

def get_scan_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT timestamp, filename, prediction, confidence FROM scan_history ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows

init_db()

# Load remedies database
try:
    with open('remedies.json', 'r', encoding='utf-8') as f:
        remedies_db = json.load(f)
except Exception as e:
    st.error(f"Error loading remedies.json: {e}")
    remedies_db = {}

# Original training classes list
classes = [
    'Pepper__bell___Bacterial_spot images',
    'Pepper__bell___healthy images',
    'Potato___Early_blight images',
    'Potato___healthy images',
    'Potato___Late_blight images',
    'Tomato_Bacterial_spot images',
    'Tomato_Early_blight images',
    'Tomato_healthy images',
    'Tomato_Late_blight images',
    'Tomato_Leaf_Mold images',
    'Tomato_Septoria_leaf_spot images',
    'Tomato_Spider_mites_Two_spotted_spider_mite images',
    'Tomato__Target_Spot images',
    'Tomato__Tomato_mosaic_virus images',
    'Tomato__Tomato_YellowLeaf__Curl_Virus images'
]

@st.cache_resource
def load_classification_model():
    model_path = 'auto_chloro_model.h5'
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        st.warning(f"Model file '{model_path}' was not found. Using simulated prediction mode for demonstration.")
        return None

model = load_classification_model()

# UI Layout
st.title("🌱 Auto-Chloro: Crop Disease Web App")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Scan Crop", "📜 History Logs", "ℹ️ About Application"])

with tab1:
    st.header("Crop Disease Diagnosis")
    st.write("Upload a clear picture of a potato, tomato, or pepper leaf to detect disease infection early.")
    
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Leaf Image", use_container_width=True)
            
        with col2:
            st.write("### Diagnostic Results")
            with st.spinner("Analyzing image..."):
                try:
                    # Preprocess Image
                    pil_img = Image.open(uploaded_file).convert('RGB')
                    resized_img = pil_img.resize((48, 48))
                    img_array = keras_image.img_to_array(resized_img)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    if model is not None:
                        # Inference
                        prediction = model.predict(img_array)
                        probs = prediction[0]
                        top_index = int(probs.argmax())
                        top_class = classes[top_index]
                        top_confidence = float(probs[top_index] * 100)
                        
                        # Extract top 3 indices
                        top_3_indices = probs.argsort()[-3:][::-1]
                    else:
                        # Simulated Fallback Mode
                        top_index = 2
                        top_class = classes[top_index]
                        top_confidence = 94.75
                        top_3_indices = [2, 4, 0]
                        probs = np.zeros(15)
                        probs[2] = 0.9475
                        probs[4] = 0.0350
                        probs[0] = 0.0175
                    
                    info = remedies_db.get(top_class, {"display_name": top_class, "remedy": "No remedy recommendation found."})
                    
                    st.success(f"**Detected:** {info['display_name']} with **{top_confidence:.2f}%** confidence.")
                    
                    # Store scan to local SQLite
                    log_scan(uploaded_file.name, info['display_name'], top_confidence)
                    
                    st.markdown("#### Actionable Remedies")
                    st.info(info['remedy'])
                    
                    # Top-3 predictions
                    st.markdown("#### Top-3 Model Candidates:")
                    for idx in top_3_indices:
                        candidate_class = classes[idx]
                        candidate_info = remedies_db.get(candidate_class, {"display_name": candidate_class})
                        candidate_confidence = probs[idx] * 100
                        st.write(f"- **{candidate_info['display_name']}**: {candidate_confidence:.2f}%")
                        st.progress(int(candidate_confidence))
                        
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

with tab2:
    st.header("Farmer Scan History Logs")
    st.write("Track past disease reports in your local database:")
    history_data = get_scan_history()
    
    if len(history_data) > 0:
        import pandas as pd
        df = pd.DataFrame(history_data, columns=["Timestamp", "Filename", "Diagnosed Disease", "Confidence Score (%)"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No scan history logs found yet. Diagnose an image above to start your database records.")

with tab3:
    st.header("About Auto-Chloro App")
    st.write("""
    Auto-Chloro is an AI-driven agricultural decision-support system initially designed for potato, tomato, and pepper crops.
    It utilizes a deep learning Convolutional Neural Network (CNN) trained on the plant leaves dataset to run real-time inference.
    
    This modified version is tailored for **Tech-A-Thon 4.0 Problem Statement 1 (Agro-Tech & Rural Development)**:
    - **Web-based Interface**: Migrated from desktop EasyGUI to a robust web portal.
    - **Confidence Scores**: Delivers calculated model output probabilities for predictions.
    - **English Remedies**: Decoupled clean English-only remedies database.
    - **History Feature**: Locally preserves log records of scans to help farmers track history.
    """)