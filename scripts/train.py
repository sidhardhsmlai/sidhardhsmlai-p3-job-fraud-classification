# Libraries
import pandas as pd 
import joblib 
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
# Imports from our custom module
from data_processing import create_preprocessor, clean_and_combine_text

# Execution Block
if __name__ == "__main__":
    # Loading data
    print("Loading data...")
    df = pd.read_csv('data/fake_job_postings.csv')

    # --- Refactored Text Cleaning ---
    df = clean_and_combine_text(df)

    # Separate X and Y
    X = df.drop('fraudulent', axis=1)
    y = df['fraudulent']
    
    # Data Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,       
        random_state=42,     
        stratify=y           
    )

    print("Test data loaded, cleaned, and split successfully.")
    
    # "Install" the Engine
    preprocessor = create_preprocessor()
    print("Preprocessor engine created.")

    # --- Baseline Model Pipeline ---
    print("Building Baseline Pipeline...")
    lr_pipeline = Pipeline(steps=[
       ('preprocessor', preprocessor),
       ('model', LogisticRegression(
           class_weight='balanced',    
           random_state=42,            
           max_iter=1000               
       ))
    ])

    # --- Advanced Model Pipeline ---
    print("Building Advanced Pipeline...")
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1     
        ))
    ])

    # --- Hyperparameter Tuning ---
    param_grid = {
        'model__n_estimators': [100, 200], 
        'model__max_depth': [50, None] 
    }

    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf_pipeline, 
        param_grid=param_grid, 
        cv=3,                  
        scoring='f1',          
        n_jobs=-1,             
        verbose=2              
    )

    # --- Run Training ---
    print("Starting Grid Search... (This will take a few minutes)")
    grid_search.fit(X_train, y_train)
    print("Grid Search complete.")

    # --- Save Model ---
    print("--- Grid Search Results ---")
    print(f"Best F1 Score: {grid_search.best_score_}")
    print(f"Best Parameters: {grid_search.best_params_}") 
    
    output_model_path = 'results/best_pipeline.pkl'
    joblib.dump(grid_search.best_estimator_, output_model_path)

    print(f"Best model saved to {output_model_path}")