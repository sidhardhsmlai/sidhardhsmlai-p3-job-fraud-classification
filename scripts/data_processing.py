# Libraries
import pandas as pd # Data handling library
from sklearn.model_selection import train_test_split # Tool to split data for testing
from sklearn.pipeline import Pipeline # Organizes processing steps
from sklearn.compose import ColumnTransformer # Applies steps to different columns
from sklearn.feature_extraction.text import TfidfVectorizer # Converts text to numbers
from sklearn.impute import SimpleImputer # Fills in missing values
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer # Custom function application
# NLTK and Regex for custom text cleaning
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
import re 

# --- FUNCTIONS ---

# Function 1: Custom Text Cleaning (The Logic)
def custom_cleaner_logic(text):
    """Performs full text cleanup on a single string."""
    # Check if input is a string
    if not isinstance(text, str):
        return ''
    
    # Remove HTML tags
    text = re.sub('<[^>]*>', '', text)
    # Remove special characters and numbers
    text = re.sub('[^a-zA-Z]', ' ', text) 
    # Remove single characters
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize and remove stop words
    words = text.split()
    # Note: stopwords must be downloaded via nltk.download('stopwords')
    try:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
    except LookupError:
        # Fallback if NLTK data isn't found
        pass
    
    # Lemmatization
    try:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in words]
    except LookupError:
        pass
    
    return ' '.join(words)

# Wrapper function to apply cleaner to a list/series
def text_cleaning_wrapper(data):
    return [custom_cleaner_logic(x) for x in data]

# Function 2: Refactored Data Combining (Fixes D.R.Y. Violation)
def clean_and_combine_text(df):
    """
    Cleans and combines the text columns into a single 'combined_text' column.
    """
    print("Running text cleaning and combining...")

    # We must fill NaNs *before* combining
    text_cols_to_combine = ['title', 'description', 'requirements']
    df[text_cols_to_combine] = df[text_cols_to_combine].fillna('')

    # Create the *single* text column that our pipeline expects
    df['combined_text'] = df[text_cols_to_combine].apply(lambda x: ' '.join(x), axis=1)

    # Return the *modified* DataFrame
    return df

# Function 3: Preprocessor Logic (The Engine Builder)
def create_preprocessor():
    # --- Define the Text Pipeline ---
    text_pipeline = Pipeline(steps=[
        # STEP 1: Apply custom cleaning row-by-row
        ('custom_clean', FunctionTransformer(text_cleaning_wrapper, validate=False)), 
        
        # STEP 2: The TfidfVectorizer
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2)))
    ])
    
    # --- Define the Categorical Pipeline ---
    categorical_pipeline = Pipeline(steps=[
        ('imputer_cat', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # --- Define the Master ColumnTransformer ---
    text_columns = 'combined_text'
    categorical_columns = ['employment_type', 'required_experience', 'has_company_logo']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_pipeline, text_columns),
            ('categorical', categorical_pipeline, categorical_columns)
        ],
        remainder='drop' 
    )

    return preprocessor

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/fake_job_postings.csv')

    # Call refactored function
    df = clean_and_combine_text(df)

    # Define X and y
    X = df.drop('fraudulent', axis=1)
    y = df['fraudulent']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y 
    )
    
    print("Test data loaded and split successfully.")

    # Execute the test
    preprocessor = create_preprocessor()
    preprocessor.fit(X_train, y_train)

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print("--- PREPROCESSOR TEST SUCCESSFUL! ---")
    print(f"Original X_train shape: {X_train.shape}")
    print(f"Transformed X_train shape: {X_train_transformed.shape}")