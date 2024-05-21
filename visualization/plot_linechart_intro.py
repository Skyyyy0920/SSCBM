import matplotlib.pyplot as plt

ratios = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01]
ratios = ratios[::-1]
task_accuracy = [3.16, 19.61, 30.36, 43.84, 58.28, 65.1, 70.09, 79.82]
concept_accuracy = [78.31, 84.73, 87.06, 89.58, 91.90, 93.20, 94.17, 96.39]

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12

fig, ax1 = plt.subplots(figsize=(10, 10))

color = 'tab:blue'
ax1.set_xlabel('Labeled Ratio (%)', fontsize=28)
ax1.set_ylabel('Task Accuracy (%)', fontsize=28)
ax1.plot(ratios, task_accuracy, marker='o', linestyle='-', color=color, label='Task Accuracy')
ax1.tick_params(axis='y', labelsize=28)
ax1.tick_params(axis='x', labelsize=28)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Concept Accuracy (%)', fontsize=28)
ax2.plot(ratios, concept_accuracy, marker='s', linestyle='--', color=color, label='Concept Accuracy')
ax2.tick_params(axis='y', labelsize=28)

plt.title('Task and Concept Accuracy (CUB)', fontsize=32)

fig.tight_layout()
fig.legend(loc='lower right', bbox_to_anchor=(0.95, 0.1), bbox_transform=ax1.transAxes, fontsize=28)

plt.savefig('intro_acc.pdf', format='pdf')

plt.show()
