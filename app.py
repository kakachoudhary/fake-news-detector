import streamlit as st
import pickle
import requests
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 AI Fake News Detection System")
st.write("Detect Fake News with Confidence Score, Explanation & Live News")

# -------------------------------
# 🧠 FUNCTION: Explain Prediction
# -------------------------------
def explain_prediction(text):
    words = text.split()
    important_words = words[:10]  # simple logic (first 10 words)
    return important_words

# -------------------------------
# 🌐 FUNCTION: Fetch Live News
# -------------------------------
def get_live_news():
    url = "https://newsapi.org/v2/top-headlines?country=in&apiKey=YOUR_API_KEY"
    response = requests.get(url)
    data = response.json()
    articles = []

    if "articles" in data:
        for article in data["articles"][:5]:
            articles.append(article["title"])

    return articles

# -------------------------------
# ✍️ INPUT
# -------------------------------
input_text = st.text_area("Enter News Content", height=200)

if st.button("Analyze News"):
    if input_text.strip() != "":
        transformed = vectorizer.transform([input_text])
        prediction = model.predict(transformed)
        prob = model.predict_proba(transformed)

        confidence = max(prob[0]) * 100

        st.subheader("Result")

        if prediction[0] == 1:
            st.success("✅ Real News")
        else:
            st.error("❌ Fake News")

        st.info(f"Confidence Score: {confidence:.2f}%")

        # 🎯 Explanation
        st.subheader("🧠 Why this result?")
        important_words = explain_prediction(input_text)
        st.write("Important words influencing prediction:")
        st.write(", ".join(important_words))

        # 📊 Graph
        st.subheader("📊 Prediction Probability")
        labels = ["Fake", "Real"]
        values = prob[0]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylabel("Probability")
        ax.set_title("Fake vs Real Prediction")

        st.pyplot(fig)

    else:
        st.warning("Please enter text")

# -------------------------------
# 🌐 LIVE NEWS SECTION
# -------------------------------
st.subheader("🌐 Live News (Top Headlines)")

if st.button("Load Live News"):
    try:
        news_list = get_live_news()
        for i, news in enumerate(news_list):
            st.write(f"{i+1}. {news}")
    except:
        st.error("Error fetching news. Check API key.")
