import streamlit as st
import joblib

# Load the trained model
model = joblib.load("model/fake_news_model.pkl")

# Streamlit app
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.markdown("Enter a news article below and check if it is **Fake** or **Real**.")

# Text input
news_text = st.text_area("Paste news article text here:")

if st.button("Check"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text to check.")
    else:
        prediction = model.predict([news_text])

        # ✅ Fix: Ensure correct mapping (0 = Fake, 1 = Real)
        if prediction[0] == 0:
            st.error("🚨 This looks like **Fake News!**")
        else:
            st.success("✅ This looks like **Real News!**")
