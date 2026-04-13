import streamlit as st
import pickle
import requests
import matplotlib.pyplot as plt

# -------------------------------
# 🔐 API KEY (Replace with yours)
# -------------------------------
API_KEY = st.secrets["NEWS_API_KEY"]

# -------------------------------
# 📦 LOAD MODEL
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------------
# 🎨 PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.markdown("<h1 style='text-align: center;'>📰 AI Fake News Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Detect Fake News with AI, Confidence Score, Graphs & Live News</p>", unsafe_allow_html=True)

# -------------------------------
# 🧠 EXPLANATION FUNCTION
# -------------------------------
def explain_prediction(text):
    words = text.split()
    return words[:10]

# -------------------------------
# 🔥 HIGHLIGHT WORDS FUNCTION
# -------------------------------
def highlight_text(text, words):
    for word in words:
        text = text.replace(word, f"<span style='color:red; font-weight:bold'>{word}</span>")
    return text

# -------------------------------
# 🌐 FETCH LIVE NEWS
# -------------------------------
def get_live_news():
    url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={API_KEY}"

    try:
        response = requests.get(url)
        data = response.json()

        news_list = []

        if data["status"] == "ok":
            for article in data["articles"][:5]:
                title = article.get("title", "No title")
                source = article.get("source", {}).get("name", "Unknown")
                news_list.append(f"{title} ({source})")
        else:
            return ["⚠️ API error or limit reached"]

        return news_list

    except Exception as e:
        return [f"Error: {str(e)}"]

# -------------------------------
# ✍️ USER INPUT
# -------------------------------
input_text = st.text_area("✍️ Enter News Content", height=200)

if st.button("🔍 Analyze News"):
    if input_text.strip() != "":
        transformed = vectorizer.transform([input_text])
        prediction = model.predict(transformed)
        prob = model.predict_proba(transformed)

        confidence = max(prob[0]) * 100

        st.markdown("---")
        st.subheader("🧾 Result")

        if prediction[0] == 1:
            st.success("✅ Real News")
        else:
            st.error("❌ Fake News")

        st.info(f"🔎 Confidence Score: {confidence:.2f}%")

        # 📊 Graph
        st.subheader("📊 Prediction Graph")
        labels = ["Fake", "Real"]
        values = prob[0]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylabel("Probability")
        ax.set_title("Fake vs Real")

        st.pyplot(fig)

        # 🧠 Explanation
        st.subheader("🧠 Important Words")
        important_words = explain_prediction(input_text)
        st.write(", ".join(important_words))

        # 🔥 Highlight Text
        st.subheader("📝 Highlighted Text")
        highlighted = highlight_text(input_text, important_words)
        st.markdown(highlighted, unsafe_allow_html=True)

    else:
        st.warning("⚠️ Please enter some text")

# -------------------------------
# 🌐 LIVE NEWS
# -------------------------------
st.markdown("---")
st.subheader("🌐 Live News Headlines")

if st.button("📰 Load Live News"):
    news_list = get_live_news()

    for i, news in enumerate(news_list):
        st.write(f"{i+1}. {news}")

# -------------------------------
# 🤖 ANALYZE LIVE NEWS
# -------------------------------
if st.button("🤖 Analyze Live News"):
    news_list = get_live_news()

    for news in news_list:
        transformed = vectorizer.transform([news])
        prediction = model.predict(transformed)

        if prediction[0] == 1:
            st.success(f"✅ {news}")
        else:
            st.error(f"❌ {news}")

# -------------------------------
# 🎉 EXTRA UI EFFECT
# -------------------------------
st.markdown("---")
st.caption("Made with ❤️ using Machine Learning & NLP")

if st.button("🎈 Celebrate"):
    st.balloons()
