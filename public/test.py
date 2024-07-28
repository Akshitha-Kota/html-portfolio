import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.filterwarnings('ignore', category=NumbaDeprecationWarning)

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import json
from lime.lime_tabular import LimeTabularExplainer
from fairlearn.metrics import MetricFrame
from responsibleai import RAIInsights, ModelTask
import dice_ml 
from econml.dml import LinearDML
from raiwidgets import ResponsibleAIDashboard

# Utility function to convert NumPy types to Python types
def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj

# Model wrapper class
class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        predictions = self.model.predict(X)
        return np.vstack((1 - predictions, predictions)).T

# Load dataset
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target, name="MedHouseVal")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow experiment
mlflow.start_run(run_name="GradientBoostingRegressor")

# Train Gradient Boosting Regressor model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Log model
mlflow.sklearn.log_model(model, "gradient-boosting-regressor")

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mlflow.log_metric("mse", mse)

# Log parameters
mlflow.log_params(model.get_params())

# Add LIME for interpretability
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['MedHouseVal'],
    mode='regression'
)
lime_explanation = explainer.explain_instance(
    data_row=X_test.values[0],
    predict_fn=model.predict
)

# Convert explanation to JSON serializable format
explanation_json = convert_numpy_types(lime_explanation.as_list())

# Log LIME explanation to MLflow
with open("lime_explanation.json", "w") as file:
    json.dump(explanation_json, file)
mlflow.log_artifact("lime_explanation.json")

# Fairlearn for fairness evaluation
sensitive_feature = X_test['AveOccup'] > X_test['AveOccup'].median()
fairness_metric = MetricFrame(metrics=mean_squared_error, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_feature)
mlflow.log_metric("fairness_metric", fairness_metric.overall)

# DICE for counterfactual explanations
model_wrapper = ModelWrapper(model)
dice_data = dice_ml.Data(dataframe=pd.concat([X_train, y_train], axis=1), continuous_features=X_train.columns.tolist(), outcome_name='MedHouseVal')
dice_model = dice_ml.Model(model=model_wrapper, backend="sklearn")
dice_explainer = dice_ml.Dice(dice_data, dice_model, method="random")
dice_exp = dice_explainer.generate_counterfactuals(X_test.iloc[:5], total_CFs=2, desired_class="opposite")

# Inspect the dice_exp object
print(dice_exp.cf_examples_list[0].final_cfs_df)

# Save counterfactuals to CSV if available
if dice_exp.cf_examples_list[0].final_cfs_df is not None:
    dice_exp_df = dice_exp.cf_examples_list[0].final_cfs_df
    dice_exp_df.to_csv("dice_counterfactuals.csv", index=False)
    mlflow.log_artifact("dice_counterfactuals.csv")
else:
    print("No counterfactual examples were generated.")

# ECONML for causal analysis
# Define the treatment and outcome
T = X_train[['AveOccup']]  # Example treatment variable
W = X_train.drop(columns=['AveOccup'])  # Controls (all other features)
Y = y_train.values  # Outcome

# Create the LinearDML estimator
estimator = LinearDML(model_y=LinearRegression(), model_t=LinearRegression(), random_state=42)
estimator.fit(Y, T, X=W)

# Estimate treatment effect
treatment_effect = estimator.effect(X_test.drop(columns=['AveOccup']).values)

mlflow.log_metric("treatment_effect_mean", np.mean(treatment_effect))

# Integrate Responsible AI Dashboard
# Combine X_train, y_train and X_test, y_test to include the target column
train_data = X_train.copy()
train_data['MedHouseVal'] = y_train

test_data = X_test.copy()
test_data['MedHouseVal'] = y_test

# Create RAIInsights object
rai_insights = RAIInsights(model=model, train=train_data,maximum_rows_for_test=20, test=test_data, target_column='MedHouseVal', task_type=ModelTask.REGRESSION)
rai_insights.explainer.add()
rai_insights.counterfactual.add(total_CFs=10, desired_range=[y_train.min(), y_train.max()])  # Specify desired range
rai_insights.causal.add(treatment_features=['AveOccup'])
rai_insights.error_analysis.add()
rai_insights.compute()

# Save RAIInsights to avoid circular reference issues
rai_insights_path = "rai_insights.json"
rai_insights.save(rai_insights_path)
mlflow.log_artifact(rai_insights_path)

# Load RAIInsights from the saved path
rai_insights = RAIInsights.load(rai_insights_path)

# Launch Responsible AI Dashboard
ResponsibleAIDashboard(rai_insights)

# End MLflow run
mlflow.end_run()
