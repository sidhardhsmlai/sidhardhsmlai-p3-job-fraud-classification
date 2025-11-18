# Libraries
import pandas as pd # Data handling library
import joblib # Tool for loading our saved model.
import matplotlib.pyplot as plt # Tool for plotting Confusion Matrix.
from sklearn.model_selection import train_test_split # Tool to split data for testing.
from sklearn.metrics import classification_report, ConfusionMatrixDisplay # function used to grade model performance and visualize results.

# --- [FIX 1] ---
# Import our "Specialist Tool" from Phase 2
# This fixes the "D.R.Y." violation
from data_processing import clean_and_combine_text 

# Execution Block
if __name__ == "__main__":

    # --- [FIX 2] ---
    # Loading data from the *correct* relative path (no '../')
    df = pd.read_csv('data/fake_job_postings.csv')

    # --- [FIX 1] ---
    # Call our "Specialist Tool" - DO NOT REPEAT LOGIC
    # This is the "Saarland-Standard" Rule 4 fix.
    df = clean_and_combine_text(df)

    # 3. Create X (features) and y (target).
    # B's logic here was 100% correct - we keep it.
    X = df.drop(['fraudulent', 'title', 'description', 'requirements'], axis=1)
    y = df['fraudulent']

    # 4. Split the data. This must be IDENTICAL to the split used in train.py.
    # B's logic here was 100% correct - we keep it.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y   # CRUCIAL for preserving the 4.8% imbalance
    )

    print("Evaluation data loaded and split successfully.")
    
    print("Loading 'race car' (best_pipeline.pkl)...")

    # --- [FIX 3] ---
    # This is the "Saarland-Standard" Rule 13 (Reproducibility) fix.
    # We *must* use a relative path, NOT an absolute "D:\" path.
    model_path = 'results/best_pipeline.pkl'
    
    # Use joblib to load the model file from the *correct* path.
    pipeline = joblib.load(model_path)

    print("Best pipeline successfully loaded.")

    # --- [YOUR 'p3a4' MISSION STARTS ON THE NEXT LINE] ---

    # --- Task 4.3.4: Make Predictions (Taking the Exam) ---
    print("Running predictions on X_test...")
    
    # 1. Create the predictions list
    y_pred = pipeline.predict(X_test)

    # --- Task 4.4.1 - 4.4.4: Generate & Save Metrics (The Report Card) ---
    print("--- Final Classification Report ---")
    
    # 2. Calculate the report (Rule 22)
    report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'])
    print(report)

    # 3. Save the report to a file (Rule 23 Evidence)
    report_path = 'results/classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")



    # --- Task 4.4.5: Generate Confusion Matrix (The Visual) ---
    print("Generating Confusion Matrix...")
    
    # 1. Create a Figure (Canvas)
    plt.figure(figsize=(8, 6))
    
    # 2. Draw the heatmap
    # from_predictions compares y_test (Truth) vs y_pred (Guess) automatically
    ConfusionMatrixDisplay.from_predictions(
        y_test, 
        y_pred, 
        display_labels=['Real', 'Fake'],
        cmap='Blues', # Makes it a blue heatmap (darker = higher number)
        values_format='d' # Displays integers (counts), not scientific notation
    )
    
    # 3. Add a title and save
    plt.title("Confusion Matrix: Real vs Fake Predictions")
    
    # Save the image
    matrix_path = 'results/confusion_matrix.png'
    plt.savefig(matrix_path)
    print(f"Confusion matrix saved to {matrix_path}")
    
    print("Evaluation complete. All results saved.")