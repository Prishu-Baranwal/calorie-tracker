import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import json
import os

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Calorie Tracker",
    page_icon="🍽️",
    layout="wide"
)

# ── Manual nutrition data (per 100g) ─────────────────────
NUTRITION_DATA = {
    'pizza'               : {'calories': 266, 'protein': 11.0},
    'hamburger'           : {'calories': 295, 'protein': 17.0},
    'sushi'               : {'calories': 150, 'protein': 6.0},
    'ramen'               : {'calories': 436, 'protein': 18.0},
    'tacos'               : {'calories': 226, 'protein': 9.0},
    'pancakes'            : {'calories': 227, 'protein': 6.0},
    'waffles'             : {'calories': 291, 'protein': 7.9},
    'donuts'              : {'calories': 452, 'protein': 4.9},
    'ice_cream'           : {'calories': 207, 'protein': 3.5},
    'cheesecake'          : {'calories': 321, 'protein': 5.5},
    'french_fries'        : {'calories': 312, 'protein': 3.4},
    'hot_dog'             : {'calories': 290, 'protein': 11.0},
    'steak'               : {'calories': 271, 'protein': 26.0},
    'caesar_salad'        : {'calories': 130, 'protein': 7.6},
    'omelette'            : {'calories': 154, 'protein': 10.0},
    'fried_rice'          : {'calories': 163, 'protein': 3.4},
    'spaghetti_bolognese' : {'calories': 162, 'protein': 8.0},
    'chocolate_cake'      : {'calories': 371, 'protein': 4.5},
    'nachos'              : {'calories': 346, 'protein': 8.7},
    'grilled_salmon'      : {'calories': 208, 'protein': 28.0},
}

SUPPORTED_FOODS = [
    'Pizza', 'Hamburger', 'Sushi', 'Ramen', 'Tacos',
    'Pancakes', 'Waffles', 'Donuts', 'Ice Cream', 'Cheesecake',
    'French Fries', 'Hot Dog', 'Steak', 'Caesar Salad', 'Omelette',
    'Fried Rice', 'Spaghetti Bolognese', 'Chocolate Cake', 'Nachos', 'Grilled Salmon'
]

# ── Load model & data ─────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        r"C:\Users\Prishu Baranwal\calorie-tracker\models\food_classifier_v2.h5"
    )

@st.cache_data
def load_labels():
    with open(r"C:\Users\Prishu Baranwal\calorie-tracker\models\class_labels_v2.json") as f:
        return json.load(f)

model        = load_model()
class_labels = load_labels()

# ── Helper functions ──────────────────────────────────────
def predict_food(image):
    img       = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds     = model.predict(img_array)
    top3      = np.argsort(preds[0])[::-1][:3]
    return [
        {"food": class_labels[str(i)], "confidence": round(float(preds[0][i]) * 100, 1)}
        for i in top3
    ]

def get_nutrition(food_name, grams=100):
    key = food_name.lower().replace(" ", "_")
    if key in NUTRITION_DATA:
        data     = NUTRITION_DATA[key]
        calories = round((data['calories'] / 100) * grams, 1)
        protein  = round((data['protein']  / 100) * grams, 1)
        return {
            "food"    : food_name.replace("_", " ").title(),
            "grams"   : grams,
            "calories": calories,
            "protein" : protein
        }
    return {
        "food"    : food_name.replace("_", " ").title(),
        "grams"   : grams,
        "calories": 0,
        "protein" : 0
    }

def health_risk(calories, protein):
    risks = []
    if calories > 2500:
        risks.append("⚠️ High Calorie Intake — Obesity Risk")
    elif calories < 1200:
        risks.append("⚠️ Too Low Calories — Deficiency Risk")
    else:
        risks.append("✅ Balanced Calorie Intake")
    if protein < 50:
        risks.append("⚠️ Low Protein Intake!")
    else:
        risks.append("✅ Adequate Protein Level")
    return risks

# ── Session state ─────────────────────────────────────────
if "daily_log"   not in st.session_state:
    st.session_state.daily_log   = []
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "nutrition"   not in st.session_state:
    st.session_state.nutrition   = None
if "analyzed"    not in st.session_state:
    st.session_state.analyzed    = False

# ── UI ────────────────────────────────────────────────────
st.title("🍽️ AI-Based Calorie & Protein Tracker")
st.markdown("Upload a food image to detect what it is and track your daily nutrition!")

# Supported foods expander
with st.expander("📋 View Supported Food Categories (20 foods)"):
    cols = st.columns(4)
    for i, food in enumerate(SUPPORTED_FOODS):
        cols[i % 4].write(f"✅ {food}")

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Upload Food Image")
    uploaded_file = st.file_uploader("Choose a food image", type=["jpg", "jpeg", "png"])
    grams = st.slider("Portion size (grams)", 50, 500, 100, step=10)

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Analyze Food"):
            with st.spinner("Analyzing your food..."):
                st.session_state.predictions = predict_food(image)
                top_food = st.session_state.predictions[0]['food']
                st.session_state.nutrition   = get_nutrition(top_food, grams)
                st.session_state.analyzed    = True

    # ── Show predictions ──────────────────────────────────
    if st.session_state.analyzed and st.session_state.predictions:
        st.subheader("🎯 Top 3 Predictions:")
        for i, p in enumerate(st.session_state.predictions):
            confidence = p['confidence']
            bar_color  = "🟢" if confidence > 70 else "🟡" if confidence > 40 else "🔴"
            st.write(f"{i+1}. {bar_color} **{p['food'].replace('_', ' ').title()}** — {confidence}% confidence")

        nutrition = st.session_state.nutrition
        st.subheader("📊 Nutrition Info:")
        c1, c2, c3 = st.columns(3)
        c1.metric("🍽️ Food",     nutrition['food'])
        c2.metric("🔥 Calories", f"{nutrition['calories']} kcal")
        c3.metric("💪 Protein",  f"{nutrition['protein']} g")

        if st.button("➕ Add to Daily Log"):
            st.session_state.daily_log.append(st.session_state.nutrition)
            st.session_state.analyzed    = False
            st.session_state.predictions = None
            st.session_state.nutrition   = None
            st.success("✅ Added to daily log!")

with col2:
    st.subheader("📅 Daily Nutrition Log")

    if st.session_state.daily_log:
        log_df = pd.DataFrame(st.session_state.daily_log)
        st.dataframe(log_df, use_container_width=True)

        total_cal  = log_df['calories'].sum()
        total_prot = log_df['protein'].sum()

        st.subheader("📈 Daily Totals:")
        m1, m2 = st.columns(2)
        m1.metric("🔥 Total Calories", f"{round(total_cal, 1)} kcal")
        m2.metric("💪 Total Protein",  f"{round(total_prot, 1)} g")

        # Progress bars
        st.markdown("**Calorie Goal (2000 kcal):**")
        st.progress(min(total_cal / 2000, 1.0))
        st.markdown("**Protein Goal (60g):**")
        st.progress(min(total_prot / 60, 1.0))

        st.subheader("🏥 Health Risk Analysis:")
        risks = health_risk(total_cal, total_prot)
        for r in risks:
            st.write(r)

        if st.button("🗑️ Clear Log"):
            st.session_state.daily_log = []
            st.rerun()
    else:
        st.info("No meals logged yet. Upload a food image and add it to your log!")

st.markdown("---")
st.markdown("🤖 Powered by MobileNetV2 + TensorFlow | **Model Accuracy: 78.62%** | Built with Streamlit")