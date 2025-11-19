import streamlit as st  # Tool for building the web application
import pandas as pd     # Creating the single-row DataFrame from user input
import joblib           # Load saved model file 
import numpy as np      # For math operations

from data_processing import clean_and_combine_text 

# Relative path to the saved pipeline.
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

# Function that runs the web application
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
        
        # --- FIXED LOGIC: Adjusted Risk Threshold ---
        
        # 1. Get the probability (The "Risk Score")
        probs = pipeline.predict_proba(input_data_cleaned)
        risk_score = probs[0][1] 

        # 2. Define our "Safety Threshold"
        # CHANGED: Lowered to 0.20 (20%). If it's 20% risky, we WARN.
        THRESHOLD = 0.20  

        # 3. Display the Results
        st.divider()
        
        # CHANGED: Use >= so 20% triggers the warning
        if risk_score >= THRESHOLD:
            st.error(f"🚨 **WARNING: HIGH RISK DETECTED**")
            st.write("This job posting has significant indicators of being fraudulent.")
        else:
            st.success(f"✅ **SAFE: Standard Job Posting**")
            st.write("This posting appears to be consistent with real jobs.")

        # 4. The "Risk Meter" (Visual Proof)
        st.write(f"**Fraud Probability Score:** {risk_score:.2%}")
        st.progress(risk_score)
        
        # Add context based on the score
        if risk_score > 0.5:
            st.caption("The model is fairly certain this is fake.")
        elif risk_score >= 0.2:
            st.caption("⚠️ The model detects suspicious patterns (Risk > 20%). Proceed with caution.")
        else:
            st.caption("The model sees no major red flags.")
            
# Run the application
if __name__ == '__main__':
    main()