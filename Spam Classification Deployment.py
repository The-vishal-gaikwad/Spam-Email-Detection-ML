import streamlit as st
import pickle
from PIL import Image
import numpy as np

# ------------------- Page config -------------------
st.set_page_config(page_title="Spam E-Mail Classification", layout="wide")

# ------------------- Custom styles -------------------
st.markdown("""
<style>
div[class*="stTextInput"] label p {
  font-size: 26px;
}
</style>
""", unsafe_allow_html=True)

# ------------------- Load Model and Vectorizer -------------------

# Use Streamlit caching so artifacts aren't reloaded on every rerun.
@st.cache_resource
def load_artifacts():
  tfidf = pickle.load(open(r"D:\Email spam detector\Spam-Email-Detection-main\Spam-Email-Detection-main\Pickle Files\feature.pkl", 'rb'))
  model = pickle.load(open(r"D:\Email spam detector\Spam-Email-Detection-main\Spam-Email-Detection-main\Pickle Files\model.pkl", 'rb'))
  return tfidf, model

tfidf, model = load_artifacts()

# ------------------- Title and Image -------------------
st.title("Spam E-Mail Classifier")
image = Image.open(r"D:\Email spam detector\Spam-Email-Detection-main\Spam-Email-Detection-main\Data Source\images.jpg")
st.image(image, use_column_width=True)

# ------------------- Input -------------------
# Use a text_area for multi-line email input
input_mail = st.text_area("Enter the Email Message", height=200)

# show model debug info (helpful while diagnosing constant predictions)
st.caption(f"Model: {type(model)} | Vectorizer: {type(tfidf)}")

# ------------------- Prediction -------------------
if st.button("Predict"):
  if not input_mail or not input_mail.strip():
    st.warning("Please enter an email message before predicting.")
  else:
    processed_mail = input_mail.lower()
    vector_input = tfidf.transform([processed_mail])
    result = model.predict(vector_input)  # result is 'ham' or 'spam'

    # Show raw prediction and probabilities (if available)
    st.write("Raw prediction:", result)
    if hasattr(model, 'predict_proba'):
      proba = model.predict_proba(vector_input)[0]
      classes = getattr(model, 'classes_', None)
      # find index of 'spam' if exists
      try:
        spam_idx = int(list(classes).index('spam')) if classes is not None else 1
      except Exception:
        spam_idx = 1
      ham_idx = 1 - spam_idx if classes is not None else 0
      st.write(f"Probability (spam): {proba[spam_idx]:.3f}")
      st.write(f"Probability (ham): {proba[ham_idx]:.3f}")

    # Show prediction clearly using appropriate styling
    label = str(result[0])
    if label == 'spam':
      st.error("This is a Spam Mail")
    else:
      st.success("This is a Ham Mail")
