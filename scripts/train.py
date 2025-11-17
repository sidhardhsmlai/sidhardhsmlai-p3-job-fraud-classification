#libraries
import pandas as pd # Data handling library
import joblib # Used to save and load the final trained model object.
from sklearn.model_selection import train_test_split # splits data into training and testing sets.
from sklearn.pipeline import Pipeline # Organizes multiple sequential processing steps into one object.
from sklearn.model_selection import GridSearchCV # Tool for finding the best model parameters (hyperparameter tuning).
from sklearn.linear_model import LogisticRegression # A classification algorithm (our baseline model).
from sklearn.ensemble import RandomForestClassifier # ensemble classification algorithm (our advanced model).
# Phase 2 Preprocessing Engine
from data_processing import create_preprocessor, clean_and_combine_text


# Execution Block
if __name__ == "__main__":
    # Loading data from relative path
    df = pd.read_csv('data/fake_job_postings.csv')

    # --- Refactored Text Cleaning (Fixes D.R.Y. Violation) ---
    # We are now *calling* our "tool" from the toolbox
    df = clean_and_combine_text(df)
    # ---

    # Separate X and Y
    # Define X, dropping the target and the now redundant individual text columns.
    X = df.drop('fraudulent', axis=1)
    # Define the target variable (y).
    y = df['fraudulent']
    
    # Data Split
    # to prevent data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,       # Uses 20% for testing.
        random_state=42,     # Ensures reproducible results.
        stratify=y           # CRUCIAL: Preserves the 4.8% imbalance in both sets.
    )

    print("Test data loaded, cleaned, and split successfully.")
    
    # "Install" the Phase 2 engine
    preprocessor = create_preprocessor()
    
    print("Preprocessor engine successfully imported and created.")

    # --- Task 3.3.2: Define the Baseline Model Pipeline (Partner A) ---
    print("Building Baseline Pipeline (Logistic Regression)...")


    # We are creating a new "conveyor belt" (Pipeline)
    lr_pipeline=Pipeline(steps=[

       # Step 1: Our "Engine" from Phase 2 
       # This is the 'preprocessor' variable B's code created for us
    ('preprocessor',preprocessor),

    # Step 2: Our "Chassis" (the model)
    # We are creating an instance of the LogisticRegression model
    ('model',LogisticRegression(
        class_weight='balanced',  # This is our Rule 30 justification
        random_state=42,        # Ensures this model is reproducibl
        max_iter=1000           # Allows the model more time to "learn"
    ))
    ])

    print("Baseline Pipeline(lr_pipeline) created successfully.")

    # --- Task 3.3.3: Define the Advanced Model Pipeline (Partner A) ---
    print("Building Advanced Pipeline (Random Forest)...")

    # We create our second "conveyor belt"
    rf_pipeline=Pipeline(steps=[
        # Step 1: Our *same* "Engine" from Phase 2
        ('preprocessor',preprocessor),

        # Step 2: Our "Race Car Chassis" (the model)
        ('model',RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1     # Uses all your computer's "brains" or #faster command

        ))
    ])
    print("Advanced Pipeline (rf_pipeline) created successfully.")

    # --- Task 3.4.1: Define the Hyperparameter "Menu" (Partner A) ---
    # [FIX]: This is now INDENTED to be *inside* the 'if' block.

    # This is a Python Dictionary (a "key: value" menu)
    # We are telling the "mechanic" what parts to test.
    param_grid = {
        # We are targeting the 'model' step of our pipeline
        # and its 'n_estimators' setting
        'model__n_estimators': [100, 200], # Test 1: 100 trees, Test 2: 200 trees

        # We are also targeting the 'max_depth' setting
        'model__max_depth': [50, None] # Test A: 50 deep, Test B: No limit
    }
    # Total combinations to test: 2 x 2 = 4 combinations

    # --- Task 3.4.2 & 3.4.3: Create the "Master Mechanic" (GridSearchCV) ---
    # [FIX]: This is also INDENTED to be *inside* the 'if' block.

    # We are creating a new "Master Mechanic" object
    # [FIX]: Corrected typo from 'grid_seach' to 'grid_search'
    grid_search = GridSearchCV(
        estimator=rf_pipeline, # The "Race Car" we want to tune
        param_grid=param_grid, # The "menu" of parts to test
        cv=3,                  # "Cross-Validation": Run each test 3 times
        scoring='f1',          # Rule 5/22: The "score" we care about
        n_jobs=-1,             # Use all my computer's "brains" to go fast
        verbose=2              # Print updates to the terminal
    )

    print("GridSearchCV 'Master Mechanic' created successfully.")

    # --- Task 3.4.4 & 3.4.5: Run the "Master Mechanic" (Partner A) ---
    print("Starting Grid Search... (This will take a few minutes)")
    
    # This is the "GO" button.
    # The grid_search object will now:
    # 1. Take our 'rf_pipeline'.
    # 2. Test Combination 1 (100 trees, 50 depth) 3 times (cv=3).
    # 3. Test Combination 2 (100 trees, None depth) 3 times.
    # 4. Test Combination 3 (200 trees, 50 depth) 3 times.
    # 5. Test Combination 4 (200 trees, None depth) 3 times.
    # 6. It will score all 12 races *only* on the 'f1' score.
    # 7. It will find the *best* combination.

    grid_search.fit(X_train,y_train)
print("Grid Search complete.")


# --- Task 3.5: Save the Final Model (Partner A) ---
print("--- Grid Search Results ---")
print(f"Best F1 Score (from cv=3): {grid_search.best_score_}")

# 3.5.3 - 3.5.5. Save the *entire* "winning car" (pipeline)
# We save it to our "Showroom" (/results)
output_model_path = 'results/best_pipeline.pkl'

# joblib.dump is the command to "shrink-wrap" our object
joblib.dump(grid_search.best_estimator_, output_model_path)

# 3.5.6. The final payoff message
print(f"(pipeline) saved to {output_model_path}")




    

   