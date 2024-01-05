import pandas as pd
import xgboost
import shap
import matplotlib.pyplot as plt

# Create DataFrame
data = {
    "WRPT": [0.7632, 0.4267, 1.6535, 2.1415, 0.3174, 0.8376, 1.4143],
    "D_m":[.408, .482, .329, .309, .437, .348, .307],
    "a_m2/m3": [190, 305, 100, 80, 482, 206, 130],
    "E": [0.74554, 0.7, 0.8, 0.81, 0.928, 0.948, 0.96],
    "size_mm": [25, 15, 38, 50, 10, 25, 38],
    "type": ['Raschig', 'Raschig', 'Raschig', 'Raschig', 'Pall', 'Pall', 'Pall']
}

target = "WRPT"
ignore = "D_m"

df = pd.DataFrame(data)
df = df.drop("type", axis=1)
df = df.drop(ignore, axis=1)

# Separate features and target
X = df.drop(target, axis=1)
y = df[target]

# Train an XGBoost model
model = xgboost.XGBRegressor()
model.fit(X, y)

# Explain the model's predictions using SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Visualize the first prediction's explanation with a waterfall plot
shap.plots.beeswarm(shap_values)

# Show the plot
plt.show()
