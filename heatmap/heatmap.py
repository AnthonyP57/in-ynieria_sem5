import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

value = 'D_m'
X = 'E'
Y= 'a_m2/m3'


# Separate data for each type
df_raschig = df[df['type'] == 'Raschig'].drop('type', axis=1)
df_pall = df[df['type'] == 'Pall'].drop('type', axis=1)

# Create heatmaps for each type
for i, (type_label, type_df) in enumerate([('Raschig', df_raschig), ('Pall', df_pall)], start=1):
    plt.figure(figsize=(9, 6))
    sns.heatmap(pd.pivot_table(type_df, values=value, index=Y, columns=X), annot=True, fmt=".2f", linewidths=.5, cmap='viridis')
    plt.title(f'Heatmap for {type_label} Type by {value}')
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.show()
