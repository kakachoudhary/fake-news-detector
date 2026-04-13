import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Custom CSS (for better UI)
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #333;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">📰 Fake News Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Check whether a news article is Real or Fake using AI</div>', unsafe_allow_html=True)

# Input box
input_text = st.text_area("✍️ Enter News Content Here:", height=200)

# Button
if st.button("🔍 Analyze News"):
    if input_text.strip() != "":
        transformed = vectorizer.transform([input_text])
        
        prediction = model.predict(transformed)
        prob = model.predict_proba(transformed)

        confidence = max(prob[0]) * 100

        st.markdown("---")

        if prediction[0] == 1:
            st.success(f"✅ This is Real News")
        else:
            st.error(f"❌ This is Fake News")

        # Confidence score
        st.info(f"🔎 Confidence Score: {confidence:.2f}%")

        # Progress bar
        st.progress(int(confidence))

    else:
        st.warning("⚠️ Please enter some text to analyze")
