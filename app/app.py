import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import json
import os

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="NutriScan AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #ffffff; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2d3748;
    }

    /* Cards */
    .nutri-card {
        background: #1e2130;
        border: 1px solid #2d3748;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .nutri-card-green {
        background: linear-gradient(135deg, #0d2b1e, #1a3d2b);
        border: 1px solid #2d6a4f;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Metric cards */
    .metric-card {
        background: #252836;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #2d3748;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #48bb78;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Food badge */
    .food-badge {
        display: inline-block;
        background: #2d3748;
        border-radius: 8px;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        font-size: 0.85rem;
        color: #a0aec0;
    }

    /* Confidence bar */
    .conf-bar-container {
        background: #2d3748;
        border-radius: 8px;
        height: 8px;
        margin-top: 4px;
    }

    .conf-bar {
        background: linear-gradient(90deg, #48bb78, #38a169);
        border-radius: 8px;
        height: 8px;
    }

    /* Progress ring */
    .progress-label {
        font-size: 0.75rem;
        color: #718096;
        margin-bottom: 4px;
    }

    /* Risk badge */
    .risk-good {
        background: #1a3d2b;
        color: #68d391;
        border: 1px solid #2d6a4f;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        display: block;
    }

    .risk-warn {
        background: #2d2000;
        color: #f6ad55;
        border: 1px solid #7b4f12;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        display: block;
    }

    /* Title */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #48bb78, #38b2ac);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .main-subtitle {
        color: #718096;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: #1e2130;
        border-radius: 12px;
        border: 2px dashed #2d3748;
        padding: 1rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #38a169, #2d9348);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.3);
    }

    /* Divider */
    hr { border-color: #2d3748; }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Slider */
    [data-testid="stSlider"] { padding: 0.5rem 0; }

    /* Expander */
    [data-testid="stExpander"] {
        background: #1e2130;
        border: 1px solid #2d3748;
        border-radius: 12px;
    }

    /* Spinner */
    .stSpinner { color: #48bb78; }

    /* Hide default streamlit header */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Nutrition data (per 100g) ─────────────────────────────
NUTRITION_DATA = {
    'pizza'               : {'calories': 266, 'protein': 11.0, 'carbs': 33.0, 'fat': 10.0},
    'hamburger'           : {'calories': 295, 'protein': 17.0, 'carbs': 24.0, 'fat': 14.0},
    'sushi'               : {'calories': 150, 'protein': 6.0,  'carbs': 28.0, 'fat': 1.0},
    'ramen'               : {'calories': 436, 'protein': 18.0, 'carbs': 60.0, 'fat': 14.0},
    'tacos'               : {'calories': 226, 'protein': 9.0,  'carbs': 21.0, 'fat': 12.0},
    'pancakes'            : {'calories': 227, 'protein': 6.0,  'carbs': 38.0, 'fat': 7.0},
    'waffles'             : {'calories': 291, 'protein': 7.9,  'carbs': 37.0, 'fat': 14.0},
    'donuts'              : {'calories': 452, 'protein': 4.9,  'carbs': 51.0, 'fat': 25.0},
    'ice_cream'           : {'calories': 207, 'protein': 3.5,  'carbs': 24.0, 'fat': 11.0},
    'cheesecake'          : {'calories': 321, 'protein': 5.5,  'carbs': 26.0, 'fat': 23.0},
    'french_fries'        : {'calories': 312, 'protein': 3.4,  'carbs': 41.0, 'fat': 15.0},
    'hot_dog'             : {'calories': 290, 'protein': 11.0, 'carbs': 20.0, 'fat': 19.0},
    'steak'               : {'calories': 271, 'protein': 26.0, 'carbs': 0.0,  'fat': 18.0},
    'caesar_salad'        : {'calories': 130, 'protein': 7.6,  'carbs': 8.0,  'fat': 9.0},
    'omelette'            : {'calories': 154, 'protein': 10.0, 'carbs': 1.0,  'fat': 12.0},
    'fried_rice'          : {'calories': 163, 'protein': 3.4,  'carbs': 28.0, 'fat': 4.0},
    'spaghetti_bolognese' : {'calories': 162, 'protein': 8.0,  'carbs': 22.0, 'fat': 5.0},
    'chocolate_cake'      : {'calories': 371, 'protein': 4.5,  'carbs': 55.0, 'fat': 15.0},
    'nachos'              : {'calories': 346, 'protein': 8.7,  'carbs': 36.0, 'fat': 19.0},
    'grilled_salmon'      : {'calories': 208, 'protein': 28.0, 'carbs': 0.0,  'fat': 10.0},
}

SUPPORTED_FOODS = list(NUTRITION_DATA.keys())

# ── Load model ────────────────────────────────────────────
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
        data = NUTRITION_DATA[key]
        return {
            "food"    : food_name.replace("_", " ").title(),
            "grams"   : grams,
            "calories": round((data['calories'] / 100) * grams, 1),
            "protein" : round((data['protein']  / 100) * grams, 1),
            "carbs"   : round((data['carbs']    / 100) * grams, 1),
            "fat"     : round((data['fat']      / 100) * grams, 1),
        }
    return {"food": food_name.replace("_", " ").title(), "grams": grams,
            "calories": 0, "protein": 0, "carbs": 0, "fat": 0}

# ── Session state ─────────────────────────────────────────
for key in ["daily_log", "predictions", "nutrition", "analyzed"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "daily_log" else None if key != "analyzed" else False

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🥗 NutriScan AI")
    st.markdown("*AI-powered food tracker*")
    st.markdown("---")

    st.markdown("### 📊 Daily Goals")
    cal_goal  = st.slider("Calorie goal (kcal)", 1200, 3500, 2000, step=100)
    prot_goal = st.slider("Protein goal (g)",    30,   200,  60,   step=5)
    st.markdown("---")

    st.markdown("### 🍽️ Supported Foods")
    for food in SUPPORTED_FOODS:
        st.markdown(f"<span class='food-badge'>{food.replace('_',' ').title()}</span>",
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Model:** MobileNetV2")
    st.markdown("**Accuracy:** 78.62%")
    st.markdown("**Categories:** 20 foods")

# ── Main UI ───────────────────────────────────────────────
st.markdown("<p class='main-title'>🍽️ NutriScan AI</p>", unsafe_allow_html=True)
st.markdown("<p class='main-subtitle'>Upload a food image to get instant nutrition info and track your daily intake</p>",
            unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

# ── LEFT COLUMN ───────────────────────────────────────────
with col1:
    st.markdown("### 📸 Scan Food")

    uploaded_file = st.file_uploader(
        "Drop a food image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    grams = st.slider("🔢 Portion size (grams)", 50, 500, 100, step=10)

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)

        if st.button("🔍 Analyze Food"):
            with st.spinner("Scanning your food..."):
                st.session_state.predictions = predict_food(image)
                top_food = st.session_state.predictions[0]['food']
                st.session_state.nutrition   = get_nutrition(top_food, grams)
                st.session_state.analyzed    = True

    # ── Predictions ───────────────────────────────────────
    if st.session_state.analyzed and st.session_state.predictions:
        st.markdown("### 🎯 Predictions")
        for i, p in enumerate(st.session_state.predictions):
            conf  = p['confidence']
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            color = "#48bb78" if conf > 70 else "#f6ad55" if conf > 40 else "#fc8181"
            st.markdown(f"""
            <div class='nutri-card' style='padding: 1rem; margin-bottom: 0.5rem;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='font-weight: 600;'>{emoji} {p['food'].replace('_',' ').title()}</span>
                    <span style='color: {color}; font-weight: 700;'>{conf}%</span>
                </div>
                <div class='conf-bar-container'>
                    <div class='conf-bar' style='width: {conf}%; background: {color};'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Nutrition info
        n = st.session_state.nutrition
        st.markdown("### 📊 Nutrition Breakdown")
        st.markdown(f"""
        <div class='nutri-card-green'>
            <h3 style='color: #68d391; margin-top: 0;'>{n['food']} — {n['grams']}g</h3>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 12px;'>
                <div class='metric-card'>
                    <div class='metric-value'>{n['calories']}</div>
                    <div class='metric-label'>🔥 Calories</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value' style='color: #63b3ed;'>{n['protein']}g</div>
                    <div class='metric-label'>💪 Protein</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value' style='color: #f6ad55;'>{n['carbs']}g</div>
                    <div class='metric-label'>🌾 Carbs</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value' style='color: #fc8181;'>{n['fat']}g</div>
                    <div class='metric-label'>🥑 Fat</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("➕ Add to Daily Log"):
            st.session_state.daily_log.append(st.session_state.nutrition)
            st.session_state.analyzed    = False
            st.session_state.predictions = None
            st.session_state.nutrition   = None
            st.success("✅ Added to your daily log!")

# ── RIGHT COLUMN ──────────────────────────────────────────
with col2:
    st.markdown("### 📅 Daily Log")

    if st.session_state.daily_log:
        log_df     = pd.DataFrame(st.session_state.daily_log)
        total_cal  = log_df['calories'].sum()
        total_prot = log_df['protein'].sum()
        total_carbs = log_df['carbs'].sum()
        total_fat  = log_df['fat'].sum()

        # Daily totals
        st.markdown(f"""
        <div class='nutri-card'>
            <h4 style='color: #a0aec0; margin-top: 0;'>TODAY'S TOTALS</h4>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 12px;'>
                <div class='metric-card'>
                    <div class='metric-value'>{round(total_cal)}</div>
                    <div class='metric-label'>🔥 Calories</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value' style='color: #63b3ed;'>{round(total_prot)}g</div>
                    <div class='metric-label'>💪 Protein</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value' style='color: #f6ad55;'>{round(total_carbs)}g</div>
                    <div class='metric-label'>🌾 Carbs</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value' style='color: #fc8181;'>{round(total_fat)}g</div>
                    <div class='metric-label'>🥑 Fat</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Progress bars
        cal_pct  = min(total_cal  / cal_goal,  1.0)
        prot_pct = min(total_prot / prot_goal, 1.0)
        cal_color  = "#48bb78" if cal_pct  < 0.85 else "#f6ad55" if cal_pct < 1.0 else "#fc8181"
        prot_color = "#48bb78" if prot_pct > 0.5  else "#f6ad55"

        st.markdown(f"""
        <div class='nutri-card'>
            <h4 style='color: #a0aec0; margin-top: 0;'>GOAL PROGRESS</h4>
            <div style='margin-bottom: 1rem;'>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: #a0aec0; font-size: 0.85rem;'>🔥 Calories</span>
                    <span style='color: {cal_color}; font-size: 0.85rem; font-weight: 600;'>{round(total_cal)} / {cal_goal} kcal</span>
                </div>
                <div class='conf-bar-container' style='margin-top: 6px;'>
                    <div class='conf-bar' style='width: {cal_pct*100:.1f}%; background: {cal_color};'></div>
                </div>
            </div>
            <div>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: #a0aec0; font-size: 0.85rem;'>💪 Protein</span>
                    <span style='color: {prot_color}; font-size: 0.85rem; font-weight: 600;'>{round(total_prot)}g / {prot_goal}g</span>
                </div>
                <div class='conf-bar-container' style='margin-top: 6px;'>
                    <div class='conf-bar' style='width: {prot_pct*100:.1f}%; background: {prot_color};'></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Health risk
        st.markdown("### 🏥 Health Analysis")
        if total_cal > cal_goal * 1.1:
            st.markdown("<span class='risk-warn'>⚠️ High calorie intake — above your daily goal</span>",
                        unsafe_allow_html=True)
        elif total_cal < cal_goal * 0.5:
            st.markdown("<span class='risk-warn'>⚠️ Too few calories consumed today</span>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<span class='risk-good'>✅ Calorie intake is on track</span>",
                        unsafe_allow_html=True)

        if total_prot < prot_goal * 0.5:
            st.markdown("<span class='risk-warn'>⚠️ Low protein — consider adding protein-rich foods</span>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<span class='risk-good'>✅ Protein intake is adequate</span>",
                        unsafe_allow_html=True)

        # Meal log table
        st.markdown("### 🍽️ Meals Logged")
        st.dataframe(
            log_df[['food', 'grams', 'calories', 'protein', 'carbs', 'fat']],
            use_container_width=True,
            hide_index=True
        )

        if st.button("🗑️ Clear Daily Log"):
            st.session_state.daily_log = []
            st.rerun()

    else:
        st.markdown("""
        <div class='nutri-card' style='text-align: center; padding: 3rem;'>
            <div style='font-size: 3rem;'>🍽️</div>
            <h3 style='color: #718096;'>No meals logged yet</h3>
            <p style='color: #4a5568;'>Upload a food image and add it to start tracking your nutrition</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #4a5568; font-size: 0.85rem;'>"
    "🤖 Powered by MobileNetV2 + TensorFlow &nbsp;|&nbsp; "
    "Model Accuracy: 78.62% &nbsp;|&nbsp; Built with Streamlit"
    "</p>",
    unsafe_allow_html=True
)