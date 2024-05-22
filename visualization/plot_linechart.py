import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.DataFrame({
    "Intervention Ratio": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "Accuracy": [i / 100 for i in [84.85, 86.33, 86.46, 89.14, 90.35, 91.69, 92.76, 93.16, 94.77, 95.44, 96.11]]
})
# Set the size and the style of seaborn plots
sns.set(rc={'figure.figsize': (12, 6)})  # Adjust the figure size to accommodate two plots side by side
sns.set_style("whitegrid")  # Changed to "whitegrid" for grid background

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, sharey=True)  # 1 row, 2 columns, share y-axis

# First plot
sns.lineplot(data=df1, x='Intervention Ratio', y='Accuracy', marker='o', markers=True, dashes=False, ax=axes[0])
axes[0].set_title('CUB', fontsize=20, fontweight='bold')
axes[0].set_xlabel('Intervention Ratio', fontsize=20, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=20, fontweight='bold')
axes[0].grid(color='black', linestyle='--', linewidth=0.5)  # Changed grid color to black
axes[0].spines['bottom'].set_color('black')
axes[0].spines['top'].set_color('black')
axes[0].spines['right'].set_color('black')
axes[0].spines['left'].set_color('black')
# Second plot (assuming the same data for demonstration, you can replace with another dataset)
df1 = pd.DataFrame({
    "Intervention Ratio": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "Accuracy": [i / 100 for i in [68.03, 75.13, 81.10, 86.31, 90.48, 91.74, 94.24, 95.99, 97.08, 97.24, 98.16]]
})
sns.lineplot(data=df1, x='Intervention Ratio', y='Accuracy', marker='o', markers=True, dashes=False, ax=axes[1])
axes[1].set_title('AwA2', fontsize=20, fontweight='bold')
axes[1].set_xlabel('Intervention Ratio', fontsize=20, fontweight='bold')
axes[1].set_ylabel('Accuracy', fontsize=20, fontweight='bold')
axes[1].grid(color='black', linestyle='--', linewidth=0.5)  # Changed grid color to black
axes[1].spines['bottom'].set_color('black')
axes[1].spines['top'].set_color('black')
axes[1].spines['right'].set_color('black')
axes[1].spines['left'].set_color('black')

plt.tight_layout()  # Adjust the layout to prevent overlapping
plt.savefig('zhexian_c-l_combined.pdf', format='pdf', dpi=600)
plt.show()
