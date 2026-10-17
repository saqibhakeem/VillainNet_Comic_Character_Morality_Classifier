import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import re

st.set_page_config(
    page_title="Comic Character Morality Predictor",
    page_icon="🦸",
    layout="wide"
)

# -------------------------------------------------------------
# 1. Load Model and Preprocessing Artifacts
# -------------------------------------------------------------
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("comic_alignment_model.keras")
    artifacts = joblib.load("pipeline_artifacts.joblib")
    return model, artifacts

try:
    model, artifacts = load_assets()
    scaler = artifacts['scaler']
    tfidf = artifacts['tfidf']
    tabular_columns = artifacts['tabular_columns']
    top_cats = artifacts['top_cats']
    year_median = artifacts['year_median']
except Exception as e:
    st.error(
        f"Error loading model artifacts: {e}. "
        "Make sure 'comic_alignment_model.keras' and 'pipeline_artifacts.joblib' are in the same directory as app.py."
    )
    st.stop()

# -------------------------------------------------------------
# 2. Feature Transformation Pipeline
# -------------------------------------------------------------
def transform_input(data: dict):
    raw_name = str(data.get("name", "")).strip()
    
    # Strip any parenthetical text so TF-IDF evaluates only the core character name
    clean_name = re.sub(r"\s*\(.*?\)", "", raw_name).strip()
    if not clean_name:
        clean_name = raw_name

    # Helper to format categorical strings identically to training
    def clean_val(val, key):
        if not val or val == "Unknown":
            v = "Unknown"
        else:
            v = re.sub(r" Characters| Eyes| Hair| Identity", "", str(val)).strip()
        allowed = top_cats.get(key, [])
        return v if v in allowed else "Other"

    identity_choice = data.get("id", "Unknown")

    row_cats = {
        'publisher': data.get("publisher", "Marvel"),
        'id': clean_val(identity_choice, 'id'),
        'sex': clean_val(data.get("sex"), 'sex'),
        'alive': clean_val(data.get("alive"), 'alive'),
        'eye': clean_val(data.get("eye"), 'eye'),
        'hair': clean_val(data.get("hair"), 'hair')
    }
    
    cat_df = pd.get_dummies(pd.DataFrame([row_cats]), drop_first=True, dtype=float)
    
    # Missingness flags
    missing_eye = 1.0 if data.get("eye") in [None, "Unknown"] else 0.0
    missing_hair = 1.0 if data.get("hair") in [None, "Unknown"] else 0.0
    missing_id = 1.0 if identity_choice in [None, "Unknown"] else 0.0
    
    # Numerical features
    raw_year = data.get("year")
    year_filled = float(raw_year) if raw_year is not None else year_median
    log_apps = float(np.log1p(max(0, data.get("appearances", 1))))
    years_active = float(max(2014.0 - year_filled, 1.0))
    apps_per_year = log_apps / years_active

    # Backwards-compatible fallback if running with an older saved model
    has_alter_ego = 1.0 if (
        re.search(r"\(.*\)", raw_name) or 
        identity_choice in ["Secret Identity", "Public Identity"]
    ) else 0.0
    name_length = float(len(raw_name))
    
    num_df = pd.DataFrame([{
        'LOG_APPEARANCES': log_apps,
        'YEAR_FILLED': year_filled,
        'YEARS_ACTIVE': years_active,
        'APPEARANCES_PER_YEAR': apps_per_year,
        'MISSING_EYE': missing_eye,
        'MISSING_HAIR': missing_hair,
        'MISSING_ID': missing_id,
        'HAS_ALTER_EGO': has_alter_ego,
        'NAME_LENGTH': name_length
    }])
    
    # Merge and align strictly to the exact columns expected by the scaler
    combined_tab = pd.concat([cat_df, num_df], axis=1)
    combined_tab = combined_tab.reindex(columns=tabular_columns, fill_value=0.0)
    
    tab_scaled = scaler.transform(combined_tab)
    text_vector = tfidf.transform([clean_name]).toarray()
    
    return np.hstack([tab_scaled, text_vector]), clean_name

# -------------------------------------------------------------
# 3. Streamlit User Interface
# -------------------------------------------------------------
st.title("🦸 Comic Character Morality Classifier")
st.markdown("Predicts whether a comic character is a **Hero (Good)** or **Villain (Bad)** based on character attributes and naming patterns[cite: 3].")

with st.form("character_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Identity & Origins")
        name = st.text_input(
            "Character Name",
            value="Spider-Man",
            help="Enter any hero or villain name (e.g., 'Captain America', 'Doctor Doom', 'Batman')"
        )
        publisher = st.selectbox("Publisher", options=["DC", "Marvel"])
        identity = st.selectbox(
            "Identity Type",
            options=["Secret Identity", "Public Identity", "No Dual Identity", "Known to Authorities", "Unknown"]
        )
        sex = st.selectbox("Sex", options=["Male Characters", "Female Characters", "Other", "Unknown"])
        alive = st.selectbox("Status", options=["Living Characters", "Deceased Characters", "Unknown"])
        
    with col2:
        st.subheader("Physical Traits & Publication")
        eye = st.selectbox(
            "Eye Color",
            options=["Blue Eyes", "Brown Eyes", "Green Eyes", "Black Eyes", "Red Eyes", "Other", "Unknown"]
        )
        hair = st.selectbox(
            "Hair Color",
            options=["Black Hair", "Brown Hair", "Blond Hair", "White Hair", "Red Hair", "Other", "Unknown"]
        )
        year = st.slider("First Appearance Year", min_value=1935, max_value=2014, value=1962)
        appearances = st.number_input("Total Comic Appearances", min_value=1, max_value=5000, value=250, step=1)
        
    submitted = st.form_submit_button("Predict Alignment", use_container_width=True)

# -------------------------------------------------------------
# 4. Prediction Output & Diagnostics
# -------------------------------------------------------------
if submitted:
    input_data = {
        "name": name,
        "publisher": publisher,
        "id": identity,
        "sex": sex,
        "alive": alive,
        "eye": eye,
        "hair": hair,
        "year": year,
        "appearances": appearances
    }
    
    X_pred, parsed_name = transform_input(input_data)
    bad_prob = float(model.predict(X_pred, verbose=0)[0][0])
    good_prob = 1.0 - bad_prob
    
    st.divider()
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        if bad_prob >= 0.50:
            st.error("### 🦹 Predicted: VILLAIN (Bad Character)")
            st.metric(label="Confidence Score", value=f"{bad_prob * 100:.1f}% Villain")
        else:
            st.success("### 🛡️ Predicted: HERO (Good Character)")
            st.metric(label="Confidence Score", value=f"{good_prob * 100:.1f}% Hero")
            
        st.progress(bad_prob, text=f"Villain Probability: {bad_prob:.1%} | Hero Probability: {good_prob:.1%}")
        
    with res_col2:
        st.markdown("#### Feature Breakdown")
        st.write(f"- **Parsed Name (TF-IDF Input):** `{parsed_name}`")
        st.write(f"- **Identity Status:** `{identity}`")
        st.write(f"- **Years Active (up to 2014):** `{max(2014 - year, 1)} years`")
        st.write(f"- **Appearance Cadence:** `{appearances / max(2014 - year, 1):.1f} issues/year`")
        st.write(f"- **Living Status:** `{alive}`")