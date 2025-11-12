# Libraries
import pandas as pd # Data handling library
from sklearn.model_selection import train_test_split # Tool to split data for testing
from sklearn.pipeline import Pipeline # Organizes processing steps
from sklearn.compose import ColumnTransformer # Applies steps to different columns
from sklearn.feature_extraction.text import TfidfVectorizer # Converts text to numbers
from sklearn.impute import SimpleImputer # Fills in missing values
from sklearn.preprocessing import OneHotEncoder # Converts categories to numbers

#making preprocessing logic importable and reusable
def create_preprocessor():
    #Partner A
    #making preprocessing logic importable and reusable
    text_pipeline = Pipeline(steps=[
        
         # Step 1: Convert text to numbers using TF-IDF
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2)))
    ])
    
    # [FIX] This line *must* be indented to be *inside* the function
    # --- Task 2.2.4: Define the Categorical Pipeline ---
    # This "conveyor belt" handles all our category columns
    categorical_pipeline = Pipeline(steps=[
        # Step 1: Fill missing categories with the most common one
        ('imputer_cat', SimpleImputer(strategy='most_frequent')),
        
        # Step 2: Convert categories (like "Full-time") to 1s and 0s
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # [FIX] All the following lines *must* also be indented

    # --- Task 2.2.5 & 2.2.6: Define the Master ColumnTransformer ---
    # This is the "master routing station" that applies the right
    # pipeline to the right columns.
    
    # These are the columns our Phase 1 EDA justified keeping
    text_columns = 'combined_text'
    categorical_columns = ['employment_type', 'required_experience', 'has_company_logo']
    
    preprocessor = ColumnTransformer(
        transformers=[
            # Apply the 'text_pipeline' to all 'text_columns'
            ('text', text_pipeline, text_columns),
            
            # Apply the 'categorical_pipeline' to all 'categorical_columns'
            ('categorical', categorical_pipeline, categorical_columns)
        ],
        
        # This is our Rule 30 Justification:
        # Drop all other columns (like salary_range, department)
        # that our Phase 1 EDA proved were unusable.
        remainder='drop' 
    )

    # --- Task 2.2.7: Return the Preprocessor ---
    # [FIX] This 'return' *must* be the last line *inside* the function
    return preprocessor


#Test Block
if __name__ == "__main__":
    # Load data from the relative path
    df = pd.read_csv('data/fake_job_postings.csv')

    # Predicts y using the data in X
    # Define features (X) by dropping the target column
    X = df.drop('fraudulent', axis=1)
    # Define the target variable (y)
    y = df['fraudulent']

        # --- MANUAL TEXT CLEANING (THE FIX) ---
    # We must fill NaNs *before* the pipeline
    # to avoid the "Wrong Mail" error.
    # --- MANUAL TEXT CLEANING & COMBINING (THE FIX) ---
    # We must fill NaNs *before* combining
    text_cols_to_combine = ['title', 'description', 'requirements']
    df[text_cols_to_combine] = df[text_cols_to_combine].fillna('')

    # --- NEW COMBINE STEP (Task 2.2.3 Fix) ---
    # Create the *single* text column that our pipeline expects
    # We use .apply() to join the text from each row
    df['combined_text'] = df[text_cols_to_combine].apply(lambda x: ' '.join(x), axis=1)
    # ---

    # Now, define X and y
    X = df.drop('fraudulent', axis=1)
    y = df['fraudulent']
    # ---
    # Split data: 80% train, 20% test
    # stratify=y preserves the 4.8% imbalance in both sets

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,      #Uses 20% for testing
        random_state=42,    #Ensures the exact same split every time for reproducible tests.
        stratify=y          #Ensures the 4.8% imbalance is preserved equally in both the train and test sets.
    )

    # confirmation message
    print("Test data loaded and split successfully.")

    # --- Task 2.3.3: Partner A's Test Execution ---
    # This is where we "turn the key" on the engine

    print("Testing the preprocessor engine...")

    # 1. Call your function to "get" the engine
    preprocessor = create_preprocessor()

    # 2. Fit the engine *ONLY* on the training data
    # This is where the imputer 'learns' the most_frequent
    # and the Tfidf 'learns' the 5,000 words.
    preprocessor.fit(X_train, y_train)

    # 3. Transform the training data
    X_train_transformed = preprocessor.transform(X_train)

    # 4. Transform the TEST data
    # We *only* .transform() here. We DO NOT .fit() again.
    # This prevents data leakage (Rule 5)
    X_test_transformed = preprocessor.transform(X_test)

    # 5. The final proof
    print("--- PREPROCESSOR TEST SUCCESSFUL! ---")
    print(f"Original X_train shape (rows, columns): {X_train.shape}")
    print(f"Transformed X_train shape (rows, features): {X_train_transformed.shape}")
    print(f"Transformed X_test shape (rows, features): {X_test_transformed.shape}")
