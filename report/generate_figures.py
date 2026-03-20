import matplotlib.pyplot as plt
import numpy as np

categories = ['Flat', 'Construction', 'Object', 'Nature', 'Sky', 'Human', 'Vehicle']
unet   = [0.1718, 0.4617, 0.0255, 0.0528, 0.0099, 0.0281, 0.0663]
deeplabv3 = [0.5210, 0.7397, 0.1284, 0.1617, 0.0390, 0.0492, 0.1416]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 3.5))
bars1 = ax.bar(x - width/2, unet,      width, label='Baseline U-Net',    color='#7bafd4')
bars2 = ax.bar(x + width/2, deeplabv3, width, label='DeepLabV3-ResNet50', color='#e07b54')

ax.set_ylabel('Dice Score')
ax.set_title('Per-Category Dice Scores under Corruptions')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 0.85)
ax.legend()
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('figures/robustness_per_category.pdf', bbox_inches='tight')
plt.savefig('figures/robustness_per_category.png', dpi=150, bbox_inches='tight')
print("Saved figures/robustness_per_category.png")
