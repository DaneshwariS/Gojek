import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from src.models.classifier import SklearnClassifier
from src.utils.config import load_config
from src.utils.guardrails import validate_evaluation_metrics
from src.utils.store import AssignmentStore

@validate_evaluation_metrics
def main():
    store = AssignmentStore()
    config = load_config()

    df = store.get_processed("transformed_dataset.csv")
    
    # Split data into train and test sets
    df_train, df_test = train_test_split(df, test_size=config["test_size"])

    # Define the parameter grid for randomized search
    param_dist = {
        'n_estimators': [100, 200, 500],   # Number of boosting rounds
        'max_depth': [3, 6, 10],           # Maximum depth of trees
        'learning_rate': [0.01, 0.1, 0.3], # Learning rate
        'subsample': [0.6, 0.8, 1.0],      # Fraction of data to train on
        'colsample_bytree': [0.6, 0.8, 1.0] # Fraction of features used per tree
    }

    # Initialize a LightGBM model
    lgb_estimator = lgb.LGBMClassifier(objective='binary', random_state=42)

    # Initialize RandomizedSearchCV to sample the parameter space randomly
    random_search = RandomizedSearchCV(estimator=lgb_estimator, param_distributions=param_dist, cv=3, n_iter=5, scoring='accuracy', n_jobs=-1)

    # Perform the randomized search on the training data
    random_search.fit(df_train[config["features"]], df_train[config["target"]])

    # Get the best estimator (model with the best hyperparameters)
    best_lgb = random_search.best_estimator_

    # Train the best model on the full training data
    model = SklearnClassifier(best_lgb, config["features"], config["target"])
    model.train(df_train)

    # Evaluate the model on the test data
    metrics = model.evaluate(df_test)

    # Save the best model and metrics
    store.put_model("saved_model.pkl", model)
    store.put_metrics("metrics.json", metrics)

    # Print the best hyperparameters
    print("Best hyperparameters:", random_search.best_params_)

if __name__ == "__main__":
    main()
