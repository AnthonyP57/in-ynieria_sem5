import pandas as pd
import seaborn as sns
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

#set resolution
dpi=100

df = pd.DataFrame(data)
sns.set_theme(style="ticks")

# Show distribution using kernel density estimation
g = sns.jointplot(data=df, x="a_m2/m3", y="D_m", hue="type", kind="kde")
g.fig.set_dpi(dpi)

# Show plot
plt.show()
