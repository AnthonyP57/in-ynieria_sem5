import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Create the DataFrame
data = {
    "WRPT": [0.7872, 0.4395, 1.7089, 2.2149, 0.3272, 0.8653, 1.463],
    "D_m": [.452, .534, .365, .342, .484, .385, .340],
    "a_m2/m3": [190, 305, 100, 80, 482, 206, 130],
    "E": [0.74554, 0.7, 0.8, 0.81, 0.928, 0.948, 0.96],
    "size_mm": [25, 15, 38, 50, 10, 25, 38],
    "type": ['Raschig', 'Raschig', 'Raschig', 'Raschig', 'Pall', 'Pall', 'Pall']
}

df = pd.DataFrame(data)
sns.set(style="whitegrid")

X = "D_m"
Y = "WRPT"

# Create a scatter plot with a single regression line for both types
sns.regplot(x=X, y=Y, data=df, scatter_kws={'s': 100}, ci=None, label='Raschig')

# Add another set of points without calculating regression
sns.scatterplot(x=X, y=Y, data=df[df['type'] == 'Pall'], color='red', s=100, label='Pall')

# Calculate and display R-squared for both types combined
X_combined = df[[X]]
y_combined = df[Y]
model_combined = LinearRegression().fit(X_combined, y_combined)
y_combined_pred = model_combined.predict(X_combined)
r2_combined = r2_score(y_combined, y_combined_pred)
plt.text(df[X].mean(), df[Y].max(), f'R2 = {r2_combined:.2f}', ha='center', va='center', color='blue')

# Set plot labels and title
plt.xlabel(X)
plt.ylabel(Y)
plt.title('Regression Plot for WRPT by D_m')

# Display legend
plt.legend()

# Show the plot
plt.show()
