import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from auto_ml_guardian.guardian import AutoMLGuardian
from loguru import logger
import warnings

# Suppress warnings for cleaner output in example
warnings.filterwarnings("ignore")

logger.remove()
logger.add(lambda msg: print(msg, end=""), colorize=True, level="INFO",
           format="<green>{time:HH:mm:ss}</green> <level>{level}</level> <level>{message}</level>")


logger.info("--- AutoML Guardian Example Usage ---")

# 1. Prepare your initial model and data
logger.info("Generating initial dataset and training a Logistic Regression model...")
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use a global variable to simulate updating a model in production
# In a real system, this would involve model serialization/deserialization,
# model registry interaction, or database updates.
current_model = LogisticRegression(solver='liblinear', random_state=42)
current_model.fit(X_train, y_train)

logger.info(f"Initial model trained. Baseline F1-score: {f1_score(y_test, current_model.predict(X_test), average='weighted'):.4f}")

# Define the predict and retrain functions that the Guardian will use
def model_predict_fn(data: np.ndarray) -> np.ndarray:
    """Predicts labels using the current model."""
    global current_model
    return current_model.predict(data)

def model_retrain_fn(new_X: np.ndarray, new_y: np.ndarray):
    """Re-trains the global model with new data."""
    global current_model
    logger.info("🤖 Guardian Executor: Re-training model with new data...")
    # For a real system, you might want to combine historical data with new_X, new_y
    # Here we'll just retrain on the new data for simplicity
    new_model = LogisticRegression(solver='liblinear', random_state=42) # Create new instance
    new_model.fit(new_X, new_y)
    current_model = new_model # Update the global model
    logger.success("✅ Guardian Executor: Model re-trained successfully.")
    # After retraining, you might want to update the Guardian's baseline_data if the model changed significantly
    # This example omits that for simplicity.


# 2. Initialize the AutoML Guardian
logger.info("\nInitializing AutoML Guardian...")
guardian = AutoMLGuardian(
    model_predict_fn=model_predict_fn,
    model_retrain_fn=model_retrain_fn,
    baseline_data=X_test, # Using a portion of test data as initial baseline
    baseline_predictions=current_model.predict(X_test),
    metric_threshold=0.75, # Set a lower F1-score threshold for degradation
    data_drift_threshold=0.01, # More sensitive data drift detection
    enable_human_in_loop=False # Set to True to prompt for approval
)
logger.info("AutoML Guardian ready.")

# 3. Simulate production data flow and monitoring
logger.info("\n--- Simulating Production Data Stream ---")

# Simulate 5 batches of incoming production data
for i in range(1, 6):
    logger.info(f"\n--- Processing Production Batch {i} ---")

    # Generate new production data
    batch_size = 100
    if i < 3:
        # First few batches: no drift, good performance
        new_X_prod, new_y_prod = make_classification(n_samples=batch_size, n_features=10, n_informative=5, n_redundant=0, random_state=42 + i)
        logger.info("Simulating normal production data.")
    else:
        # Introduce data drift and potential concept drift after batch 2
        logger.warning("Simulating data drift and potential concept drift!")
        # Shift some feature distributions
        drifted_X_part, drifted_y_part = make_classification(n_samples=int(batch_size/2), n_features=10, n_informative=5, n_redundant=0, random_state=100 + i)
        drifted_X_part[:, 0] += np.random.normal(0, 5, size=int(batch_size/2)) # Shift feature 0 significantly
        drifted_X_part[:, 2] += np.random.normal(0, 3, size=int(batch_size/2)) # Shift feature 2 moderately

        # Introduce some 'new' data that aligns differently with labels (concept drift proxy)
        new_concept_X, new_concept_y = make_classification(n_samples=int(batch_size/2), n_features=10, n_informative=5, n_redundant=0, random_state=200 + i)
        new_concept_X[:, 4] += np.random.normal(0, 4, size=int(batch_size/2)) # Shift feature 4 for new concept

        new_X_prod = np.vstack((drifted_X_part, new_concept_X))
        new_y_prod = np.concatenate((drifted_y_part, new_concept_y))


    # In a real scenario, actual_labels might arrive with a delay.
    # For this example, we assume they are immediately available for performance evaluation.
    guardian.monitor(new_X_prod, new_y_prod)

    # After monitoring, show current model performance (assuming labels are available)
    current_preds = current_model.predict(new_X_prod)
    current_f1 = f1_score(new_y_prod, current_preds, average='weighted')
    logger.info(f"Current Model F1-score on batch {i}: {current_f1:.4f}")

    time.sleep(1) # Simulate some processing time

logger.info("\n--- Simulation Complete ---")
logger.info("The AutoML Guardian has demonstrated its ability to monitor, diagnose, and adapt.")
logger.info("Check the log messages above for agent interactions and actions taken.")
