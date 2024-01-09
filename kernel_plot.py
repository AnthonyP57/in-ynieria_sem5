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

#set resolution
dpi=100

df = pd.DataFrame(data)
sns.set_theme(style="ticks")

# Show distribution using kernel density estimation
g = sns.jointplot(data=df, x="a_m2/m3", y="D_m", hue="type", kind="kde")
g.fig.set_dpi(dpi)

# Show plot
plt.show()
