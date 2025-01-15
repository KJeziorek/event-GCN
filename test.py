import matplotlib.pyplot as plt

# Example input data
model_params = [5200, 10600, 20200, 4500, 300000, 360000, 360000, 2030000, 3690000, 5100000, 4930000]  # Number of parameters (log-scale recommended)
model_accuracies = [0.91, 0.919, 0.925, 0.878, 0.945, 0.931, 0.915, 0.927, 0.944, 0.914, 0.902]  # Accuracies
is_gcn = [True, True, True, True, True, True, True, False, False, False, False, False]  # GCN or not
is_hardware_accelerated = [True, True, True, True, False, False, False, False, False, False, False, False, False]  # Hardware accelerated or not
highlight_models = [True, True, True, False, False, False, False, False, False, False, False]  # Highlight specific models
model_names = ["EFGCN-S", "EFGCN-B", "EFGCN-L", "EvGNN", "AEGNN", "EvS-S", "NvS-S", "YOLE", "AsyNet", "RG-CNN", "G-CNN"]  # Model names
text_offsets = [(0.01, 0.005), (0.01, 0.005), (0.01, 0.005), (0.01, 0.005), (0.01, -0.005), (0.01, 0.005), (0.01, 0.005), (0.01, 0.005), (0.01, -0.005), (0.01, 0.005), (0.01, 0.005)]  # Text offsets for each model

# Initialize the plot
plt.figure(figsize=(10, 6))

# Plot each model
for params, acc, gcn, hw_acc, highlight, name, offset in zip(model_params, model_accuracies, is_gcn, is_hardware_accelerated, highlight_models, model_names, text_offsets):
    marker = "*" if gcn else "^"  # Star for GCN, triangle for non-GCN
    fill_style = "full" if hw_acc else "none"  # Filled for hardware-accelerated, unfilled otherwise
    color = "red" if highlight else "blue"  # Highlighted models in red, others in blue

    plt.scatter(params, acc, marker=marker, facecolors=color if hw_acc else "none", edgecolors=color, s=300, label=None, linewidths=1.5)  # Plot the model
    plt.text(params * (1 + offset[0]), acc * (1 + offset[1]), name, fontsize=15, ha='center', va='center')  # Add model names with offsets


# Add legend for markers and styles
legend_elements = [
    plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='none', markersize=10, label='GCN Non-Hardware', markeredgecolor='black'),
    plt.Line2D([0], [0], marker='^', color='none', markerfacecolor='none', markersize=10, label='Non-GCN Non-Hardware', markeredgecolor='black'),
    plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='black', markersize=10, label='GCN Hardware', markeredgecolor='black'),
    # plt.Line2D([0], [0], marker='^', color='none', markerfacecolor='black', markersize=10, label='Non-GCN Hardware', markeredgecolor='black'),
]

plt.legend(handles=legend_elements, loc='best', fontsize=15)

# Customize plot
plt.xscale('log')  # Log-scale for parameters
plt.xlabel('Number of Parameters', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.title('Accuracy vs Number of Parameters', fontsize=14)
plt.grid(False, which="both", linestyle="--", linewidth=0.5)


# Show plot
plt.show()