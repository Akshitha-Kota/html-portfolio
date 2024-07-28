import matplotlib.pyplot as plt
from responsibleai import RAIInsights, ModelTask
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import mlflow
import mlflow.sklearn
import pandas as pd

# Load dataset
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target, name="MedHouseVal")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting Regressor model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Combine X_train, y_train and X_test, y_test to include the target column
train_data = X_train.copy()
train_data['MedHouseVal'] = y_train

test_data = X_test.copy()
test_data['MedHouseVal'] = y_test

# Create RAIInsights object
rai_insights = RAIInsights(model=model, train=train_data, test=test_data, target_column='MedHouseVal', task_type=ModelTask.REGRESSION)
rai_insights.explainer.add()
rai_insights.error_analysis.add()
rai_insights.compute()

# Start MLflow run
mlflow.start_run(run_name="GradientBoostingRegressor_RAI")

# Extract feature importances from the model
feature_importances = model.feature_importances_

# Save feature importance plot
plt.figure(figsize=(10, 6))
plt.barh(X_train.columns, feature_importances)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance Plot")
plt.savefig("feature_importance.png")
plt.close()

# Log feature importance plot to MLflow
mlflow.log_artifact("feature_importance.png")

# Handle error analysis results
error_analysis_results = rai_insights.error_analysis.get()

for idx, error_analysis in enumerate(error_analysis_results):
    error_analysis_path = f"error_analysis_{idx}.json"
    with open(error_analysis_path, "w") as f:
        f.write(str(error_analysis))  # Convert to string or appropriate format
    mlflow.log_artifact(error_analysis_path)

# End MLflow run
mlflow.end_run()
