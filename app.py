import streamlit as st
import pickle
import requests
import matplotlib.pyplot as plt

# -------------------------------
# 🔐 API KEY
# -------------------------------
API_KEY = st.secrets["NEWS_API_KEY"]

# -------------------------------
# 📦 LOAD MODEL
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------------
# 🎨 UI
# -------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 AI Fake News Detection")
st.write("Fake News Detection with Confidence + Live News")

# -------------------------------
# 🧠 FUNCTIONS
# -------------------------------
def explain_prediction(text):
    return text.split()[:10]

def highlight_text(text, words):
    for word in words:
        text = text.replace(word, f"<span style='color:red'>{word}</span>")
    return text

def get_live_news():
    url = f"https://newsapi.org/v2/everything?q=india&apiKey={API_KEY}"

    try:
        response = requests.get(url)
        data = response.json()

        articles = []

        if data.get("status") == "ok":
            for article in data["articles"][:5]:
                title = article.get("title", "No title")
                source = article.get("source", {}).get("name", "Unknown")
                articles.append(f"{title} ({source})")
        else:
            return [f"❌ API Error: {data.get('message')}"]

        return articles

    except Exception as e:
        return [f"Error: {str(e)}"]

# -------------------------------
# ✍️ INPUT ANALYSIS
# -------------------------------
input_text = st.text_area("Enter News Content")

if st.button("🔍 Analyze News", key="analyze_text"):
    if input_text.strip():
        X = vectorizer.transform([input_text])
        pred = model.predict(X)
        prob = model.predict_proba(X)

        confidence = max(prob[0]) * 100

        if pred[0] == 1:
            st.success("✅ Real News")
        else:
            st.error("❌ Fake News")

        st.info(f"Confidence: {confidence:.2f}%")

        # Graph
        fig, ax = plt.subplots()
        ax.bar(["Fake", "Real"], prob[0])
        st.pyplot(fig)

        # Explanation
        words = explain_prediction(input_text)
        st.write("Important words:", ", ".join(words))

        highlighted = highlight_text(input_text, words)
        st.markdown(highlighted, unsafe_allow_html=True)

# -------------------------------
# 🌐 LIVE NEWS
# -------------------------------
st.markdown("---")
st.subheader("🌐 Live News")

if st.button("📰 Load Live News", key="load_news"):
    news_list = get_live_news()

    if "Error" in news_list[0]:
        st.error(news_list[0])
    else:
        for i, news in enumerate(news_list):
            st.write(f"{i+1}. {news}")

# -------------------------------
# 🤖 ANALYZE LIVE NEWS
# -------------------------------
if st.button("🤖 Analyze Live News", key="analyze_live"):
    news_list = get_live_news()

    if "Error" in news_list[0]:
        st.error(news_list[0])
    else:
        for news in news_list:
            X = vectorizer.transform([news])
            pred = model.predict(X)

            if pred[0] == 1:
                st.success(f"✅ {news}")
            else:
                st.error(f"❌ {news}")

# -------------------------------
# 🎉 EXTRA
# -------------------------------
if st.button("🎈 Celebrate", key="celebrate"):
    st.balloons()
