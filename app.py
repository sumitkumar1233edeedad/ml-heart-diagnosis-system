import streamlit as st
import numpy as np
import skops.io as sko
import os
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="HeartCare AI Pro | Dark",
    page_icon="❤️",
    layout="wide"
)

# ---------------- DARK THEME CUSTOM CSS ----------------
st.markdown("""
    <style>
    /* Force Dark Background */
    .stApp {
        background-color: #0e1117;
        background-image: radial-gradient(circle at 2px 2px, #1d2129 1px, transparent 0);
        background-size: 40px 40px;
        color: #ffffff;
    }

    /* Sidebar Darkening */
    section[data-testid="stSidebar"] {
        background-color: #0a0c10 !important;
        border-right: 1px solid #1e2227;
    }

    /* Glassmorphism Cards */
    div[data-testid="stVerticalBlock"] > div:has(div.stSlider) {
        background-color: #161b22;
        padding: 25px;
        border-radius: 20px;
        border: 1px solid #30363d;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        margin-bottom: 15px;
    }

    /* Typography */
    .main-title {
        background: linear-gradient(to right, #ff4b2b, #ff416c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 900;
        font-size: 3.5rem;
        letter-spacing: -1px;
    }

    /* Input Labels */
    label, p, .stMarkdown {
        color: #c9d1d9 !important;
    }

    /* Glowing Button */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 4em;
        background: #e63946;
        color: white !important;
        font-weight: bold;
        font-size: 1.1rem;
        border: none;
        transition: 0.4s ease;
        box-shadow: 0 0 15px rgba(230, 57, 70, 0.4);
    }
    .stButton>button:hover {
        background: #ff4d4d;
        box-shadow: 0 0 25px rgba(230, 57, 70, 0.7);
        transform: translateY(-2px);
    }

    /* Results Card */
    .result-container {
        background: #0d1117;
        padding: 30px;
        border-radius: 20px;
        border: 2px solid #30363d;
        text-align: center;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #161b22;
        border-radius: 10px 10px 0 0;
        color: white;
        padding: 0 30px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e63946 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>❤️ HeartCare</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=120)
    st.divider()
    st.markdown("### 🛠️ System Status")
    st.success("Model: heart_model.skops")
    st.info("Environment: Production v2.0")
    st.divider()
    st.write("👨‍💻 **Dev:** Vanshuu | BCA")

# ---------------- HEADER ----------------
st.markdown("<h1 class='main-title'>HeartCare AI Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#8b949e;'>Precision Medical Intelligence System</p>", unsafe_allow_html=True)

# ---------------- INPUT STRUCTURE ----------------
tab1, tab2 = st.tabs(["📊 BIOMETRICS", "🩺 DIAGNOSTICS"])

with tab1:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("#### 👤 Demographics")
        age = st.slider("Patient Age", 18, 100, 45)
        sex = st.segmented_control("Assigned Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    
    with col2:
        st.markdown("#### 🩸 Vital Signs")
        trestbps = st.select_slider("Resting Blood Pressure", options=list(range(80, 201)), value=120)
        chol = st.select_slider("Serum Cholestoral", options=list(range(100, 501)), value=210)
        fbs = st.pills("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "Normal" if x == 0 else "Elevated")

with tab2:
    col3, col4 = st.columns(2, gap="large")
    with col3:
        st.markdown("#### 💓 Cardiac Data")
        thalach = st.slider("Max Heart Rate (BPM)", 60, 220, 155)
        cp = st.selectbox("Chest Pain Category", options=[0, 1, 2, 3], 
                          format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x])
        restecg = st.pills("Rest ECG", [0, 1, 2])

    with col4:
        st.markdown("#### 📉 Stress Test Analysis")
        oldpeak = st.slider("ST Depression Score", 0.0, 6.0, 1.2, 0.1)
        exang = st.pills("Exercise Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        ca = st.select_slider("Vessels (Fluoroscopy)", options=[0, 1, 2, 3])
        thal = st.selectbox("Thal Assessment", [1, 2, 3], 
                            format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1])
        # Added hidden slope value from previous logic to match model
        slope = 1 

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_my_model():
    try:
        model_path = os.path.join(os.getcwd(), 'models', 'heart_model.skops')
        return sko.load(model_path)
    except:
        return None

model = load_my_model()

# ---------------- PREDICTION LOGIC ----------------
st.markdown("<br>", unsafe_allow_html=True)
if st.button("⚡ ANALYZE CARDIAC RISK"):
    if model is None:
        st.error("Model file not found. Please check your /models folder.")
    else:
        with st.status("🔄 Scanning Biological Markers...", expanded=True) as status:
            time.sleep(1)
            st.write("Computing probability gradients...")
            time.sleep(1)
            
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                    thalach, exang, oldpeak, slope, ca, thal]])
            
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # UI Results
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if prediction[0] == 1:
                st.markdown(f"""
                    <div class="result-container" style="border-color: #ff4b2b;">
                        <h2 style="color: #ff4b2b;">⚠️ CRITICAL RISK</h2>
                        <h1 style="color: white; margin: 0;">{(probability*100):.1f}%</h1>
                        <p style="color: #8b949e;">Probability of Cardiovascular Disease</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-container" style="border-color: #00ff88;">
                        <h2 style="color: #00ff88;">✅ STABLE</h2>
                        <h1 style="color: white; margin: 0;">{(probability*100):.1f}%</h1>
                        <p style="color: #8b949e;">Low Risk Profile Detected</p>
                    </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("### Risk Diagnostics")
            st.progress(probability)
            st.write(f"The model is **{abs(0.5-probability)*200:.1f}%** confident in this classification.")
            if probability > 0.5:
                st.warning("Recommendation: Patient should undergo an immediate Stress Echo.")
            else:
                st.success("Recommendation: Continue annual screenings and standard diet.")

# ---------------- FOOTER ----------------
st.markdown(f"""
    <div style="margin-top: 50px; text-align: center; padding: 20px; border-top: 1px solid #30363d; color: #484f58;">
        <p><b>HEARTCARE AI</b> v2.0 • 2026 BCA Project • Vanshuu</p>
    </div>
""", unsafe_allow_html=True)