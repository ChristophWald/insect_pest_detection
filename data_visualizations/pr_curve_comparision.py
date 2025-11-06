import matplotlib.pyplot as plt
import os
import json

def plot_pr_curves_multiple_models(model_numbers, base_metrics_path, class_name):
    """
    Plot PR curves and best points for a specific class across multiple models.
    
    Args:
        model_numbers: list of model numbers as strings or ints
        base_metrics_path: base path where each model's metrics folder exists
        class_name: name of the class to plot
    """
    plt.figure(figsize=(8, 6))
    
    for model_number in model_numbers:
        model_folder = os.path.join(base_metrics_path, f"train{model_number}")
        pr_path = os.path.join(model_folder, "pr_results.json")
        best_points_path = os.path.join(model_folder, "operating_points.json")
        
        if not os.path.exists(pr_path) or not os.path.exists(best_points_path):
            print(f"⚠️ Skipping model {model_number} — missing pr_results or operating_points.")
            continue
        
        # Load PR results and best points
        with open(pr_path, "r") as f:
            pr_results = json.load(f)
        with open(best_points_path, "r") as f:
            best_points = json.load(f)
        
        # Check if class exists
        if class_name not in pr_results or class_name not in best_points:
            print(f"⚠️ Model {model_number} does not contain class '{class_name}'.")
            continue
        
        # Plot PR curve
        data = pr_results[class_name]
        plt.plot(data["recall"], data["precision"], label=f"Model {model_number}")
        
        # Highlight best point
        bp = best_points[class_name]
        plt.scatter(bp["recall"], bp["precision"], marker='o', s=50, edgecolors='k', label=f"Best Model {model_number}")
    
    # Highlight target region (optional)
    plt.fill_betweenx([0.9, 1.0], 0.8, 1.0, color='gray', alpha=0.2)
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curves for class '{class_name}' across models")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0.5, 1)
    plt.savefig(f"pr_comp_{class_name}.png")




model_numbers = ["13", "14", "15", "16"]
base_metrics_path = "/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/metrics"
class_name = "leaf miner flies"

plot_pr_curves_multiple_models(model_numbers, base_metrics_path, class_name)
class_name = "fungus gnats"
plot_pr_curves_multiple_models(model_numbers, base_metrics_path, class_name)
class_name = "whiteflies"
plot_pr_curves_multiple_models(model_numbers, base_metrics_path, class_name)
