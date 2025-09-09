import streamlit as st
import pdfplumber
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Set page config
st.set_page_config(
    page_title="Smart Resume Matcher",
    page_icon="🧠",
    layout="wide"
)
st.title("Smart Resume Matcher")
st.markdown(
    """
    ## Purpose
    This app helps job seekers and recruiters evaluate how well a resume aligns with a specific job description.   
    By uploading both documents, users receive an instant match score and prediction powered by machine learning — helping candidates tailor their resumes and recruiters filter 
    applicants more effectively.
    """)
    
st.title(" How It Works")    
st.markdown(
    """
Upload PDFs: Users upload a resume and a job description in PDF format.

Text Extraction: The app extracts and cleans text using pdfplumber and regex.

ML Model: A logistic regression model trained on sample resume–JD pairs predicts match quality.

TF-IDF Vectorization: Text is converted into numerical features using TF-IDF.

Prediction: The model outputs a match probability and verdict (Good Match / Poor Match).

User Feedback: The app displays the result with intuitive labels and confidence scores.""")
st.title(" Features") 
st.markdown(
    """
Upload and analyze resume vs job description

Real-time match prediction using ML

Confidence score for transparency

No need for external model files (self-contained training)

Simple, clean UI for non-technical users
    """)

st.title(" Why Use This Tool?")

st.markdown(
    """🔍 Precision Matching.
    """)
st.markdown(
    """⏱️ Time-Saving.
    """)
st.markdown(
    """
    📈 Career Optimization.
    """)


st.image('https://your-direct-image-url.jpg',
         caption="Automated Resume Analysis Made Simple",
         use_container_width=True)

st.sidebar.title("Navigate")
st.sidebar.write("Use the navigation panel to explore different features of the app.")


st.title("""Created by Group D:""")
st.markdown("Tripti Yadav (905)")
st.markdown("Sakshi Indulkar (910)")
st.markdown("Shruthi Thootey (922)")
st.markdown("Mangesh Patel (901)")
st.markdown("Yash Kadam (914)")
st.markdown("Varsha Gupta (917)")
st.markdown("Maheshwari Yadav (934)")
 

# Apply Light Lavender and Light Mint Green Gradient background with black text
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to bottom, #D1C4E9, #A5D6A7);  /* Light Lavender to Light Mint Green Gradient */
            color: black;  /* Black text for better readability on a light background */
        }
        .block-container {
            background: linear-gradient(to bottom, #D1C4E9, #A5D6A7);  /* Keep the content container background same */
            color: black;  /* Black text in content area */
        }
        .css-18e3t6p {
            color: black;  /* Ensure any other texts are also black */
        }
    </style>
    """, unsafe_allow_html=True
)
# Sample training data
def get_training_data():
    data = {
        "resume_text": [
            "Experienced Python developer with Selenium and JIRA",
            "Marketing specialist with SEO background",
            "QA engineer with 3 years in manual testing",
            "Java backend developer"
        ],
        "jd_text": [
            "Looking for Python tester with Selenium experience",
            "Hiring software engineer with Python and Git",
            "Manual testing role requiring 2+ years experience",
            "Frontend React developer needed"
        ],
        "label": [1, 0, 1, 0]
    }
    return pd.DataFrame(data)

# Train Random Forest model
def train_model():
    df = get_training_data()
    combined = df["resume_text"] + " " + df["jd_text"]
    labels = df["label"]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(combined)
    model = RandomForestClassifier()
    model.fit(X, labels)
    return model, vectorizer

# PDF text extraction
def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Text cleaning
def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()

# App UI
st.title("🧠 Smart Resume Matcher")
st.markdown("Upload your resume and job description to check match quality using Random Forest.")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

if resume_file and jd_file:
    with st.spinner("Analyzing with Random Forest model..."):
        resume_text = clean_text(extract_text(resume_file))
        jd_text = clean_text(extract_text(jd_file))
        combined_input = resume_text + " " + jd_text

        model, vectorizer = train_model()
        X_input = vectorizer.transform([combined_input])
        prediction = model.predict(X_input)[0]

        try:
            probability = model.predict_proba(X_input)[0][1]
            match_percent = round(probability * 100, 2)
            st.success(f"🔍 Match Score: {match_percent}%")
        except:
            match_percent = None
            st.success("🔍 Prediction made (probability not available)")

    if match_percent is not None:
        if match_percent >= 75:
            st.markdown("### ✅ Strong Match!")
        elif match_percent >= 50:
            st.markdown("### ⚠️ Moderate Match — Consider Improving Your Resume")
        else:
            st.markdown("### ❌ Weak Match — Resume May Not Align Well")
    else:
        if prediction == 1:
            st.markdown("### ✅ Good Match!")
        else:
            st.markdown("### ❌ Poor Match — Consider Revising Your Resume")
else:
    st.info("Please upload both files to begin.")



