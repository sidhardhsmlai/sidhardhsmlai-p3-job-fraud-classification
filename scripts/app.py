import streamlit as st  # tool for building the web application
import pandas as pd     # creating the single-row DataFrame from user input
import joblib           # load saved model file 
import numpy as np      # For math operations

from data_processing import clean_and_combine_text 

# relative path to the saved pipeline.
MODEL_PATH = 'results/best_pipeline.pkl'

# Function to load and reuse the model
@st.cache_resource
def load_pipeline():
    # Load the Engine (the trained pipeline).
    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {MODEL_PATH}. Ensure p3a3 was completed correctly.")
        return None

# function that runs the web application
def main():
    # Load the Engine
    pipeline = load_pipeline()
    if pipeline is None:
        return

    # UI Requirements
    st.title("🛡 Fraudulent Job Posting Detector")
    st.markdown("### The Captain's Bridge: Analyze a Job Posting")

    # Get user input for the three critical features
    title = st.text_input("Job Title", help="e.g., Senior Data Scientist")
    description = st.text_area("Job Description", help="Paste the full job responsibilities here.")
    requirements = st.text_area("Requirements/Qualifications", help="List the required skills and degrees.")
    
    # Prediction Button
    if st.button("ANALYZE JOB"):
        if not title and not description and not requirements:
             st.warning("Please enter some text in at least one field to analyze.")
             return

        # Processing Logic
        # Create a single-row DataFrame from user input
        input_data = pd.DataFrame({
            'title': [title],
            'description': [description],
            'requirements': [requirements],
            
            # Placeholder columns for the remaining features the model expects.
            # The ColumnTransformer will process or drop these as needed.
            'location': ['US, NY, New York'],
            'department': [None], 
            'salary_range': [None],
            'company_profile': [''],
            'benefits': [''],
            'telecommuting': [0],
            'has_company_logo': [1],
            'has_questions': [1],
            'employment_type': ['Full-time'],
            'required_experience': ['Mid-Senior level'],
            'required_education': ['Bachelor\'s Degree'],
            'industry': ['Technology'],
            'function': ['Engineering']
        })

        # Pass the DataFrame through the clean_and_combine_text utility
        input_data_cleaned = clean_and_combine_text(input_data)
        
        # Making the Prediction
        prediction = pipeline.predict(input_data_cleaned)[0]
        
        # Get the probability score
        proba = pipeline.predict_proba(input_data_cleaned)
        confidence = proba[0][prediction] * 100
        
        # Output Display
        if prediction == 0:
            st.success("✅ Prediction: REAL JOB POSTING")
            st.write(f"Confidence: **{confidence:.2f}%**")
        else:
            st.error("🚨 Prediction: FAKE/FRAUDULENT JOB POSTING")
            st.write(f"Confidence: **{confidence:.2f}%**")
            
# Run the application
if __name__ == '__main__':
    main()