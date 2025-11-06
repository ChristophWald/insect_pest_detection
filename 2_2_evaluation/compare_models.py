import matplotlib.pyplot as plt
import os
import json

def plot_pr_curves(pr_results, best_points=None, second_points=None, base_output_path=None): 
    """
    Plot PR curves for each class and optionally highlight best points.
    
    Args:
        pr_results: dict from compute_pr_curves_with_all
        best_points: dict of points to highlight with circles (optional)
        second_points: dict of points to highlight with squares (optional)
        base_output_path: folder to save the plot (optional)
    """
    plt.figure(figsize=(8, 6)) 
    
    # Plot each class curve
    for cls_name, data in pr_results.items(): 
        plt.plot(data["recall"], data["precision"], label=cls_name)
        
        # Highlight best_points (circles)
        if best_points and cls_name in best_points:
            bp = best_points[cls_name]
            plt.scatter(bp["recall"], bp["precision"], marker='o', s=20, edgecolors='k')
        
        # Highlight second_points (squares)
        if second_points and cls_name in second_points:
            sp = second_points[cls_name]
            plt.scatter(sp["recall"], sp["precision"], marker='s', s=20, edgecolors='k')
    
    # Highlight target region
    plt.fill_betweenx([0.9, 1.0], 0.8, 1.0, color='gray', alpha=0.3)
    #plt.text(0.805, 0.905, 'Precision ≥ 0.9\nRecall ≥ 0.8', fontsize=9, color='gray')
    
    # Labels and styling
    plt.xlabel("Recall") 
    plt.ylabel("Precision") 
    plt.title("Precision-Recall Curve") 
    plt.legend(loc="lower left", fontsize=8) 
    plt.grid(True) 

    # Save plot if path provided
    if base_output_path:
        os.makedirs(base_output_path, exist_ok=True)
        plt.savefig(os.path.join(base_output_path, "pr_curve_compare.jpg"), dpi=300, bbox_inches="tight")
    
    plt.show()
    plt.close()

import numpy as np

def compare_best_point_sets_per_class(best_points_a, best_points_b, prec_thresh=0.9, rec_thresh=0.8, verbose=True, return_details=True):
    """
    Compare two sets of best_points (outputs from find_best_pr_points).

    Now also determines the better model per class.

    Logic:
      1. Prefer results with more points inside the "good zone".
      2. If equal, prefer smaller average distance to the rectangle.
      3. If still equal, prefer higher average F1.
    """

    def distance_to_rectangle(prec, rec, prec_thresh, rec_thresh):
        dx = max(0.0, prec_thresh - prec)
        dy = max(0.0, rec_thresh - rec)
        return np.sqrt(dx**2 + dy**2)

    def analyze_set(best_points):
        precisions = np.array([p["precision"] for p in best_points.values()])
        recalls = np.array([p["recall"] for p in best_points.values()])
        f1s = np.array([p["f1"] for p in best_points.values()])

        inside_mask = (precisions >= prec_thresh) & (recalls >= rec_thresh)
        num_inside = np.sum(inside_mask)
        avg_f1 = np.mean(f1s)
        dists = np.array([
            distance_to_rectangle(p, r, prec_thresh, rec_thresh)
            for p, r in zip(precisions, recalls)
        ])
        avg_dist = np.mean(dists)

        # Per-class details
        details = {
            cls_name: {
                "precision": p["precision"],
                "recall": p["recall"],
                "f1": p["f1"],
                "dist_to_zone": distance_to_rectangle(p["precision"], p["recall"], prec_thresh, rec_thresh),
                "inside_zone": (p["precision"] >= prec_thresh and p["recall"] >= rec_thresh),
            }
            for cls_name, p in best_points.items()
        }

        return num_inside, avg_f1, avg_dist, details

    # Analyze both sets
    num_in_a, f1_a, dist_a, details_a = analyze_set(best_points_a)
    num_in_b, f1_b, dist_b, details_b = analyze_set(best_points_b)

    # === OVERALL COMPARISON ===
    if verbose:
        print("=== Overall Comparison ===")
        print(f"Thresholds: precision ≥ {prec_thresh}, recall ≥ {rec_thresh}\n")
        print(f"Set A → inside: {num_in_a}, avg_dist: {dist_a:.4f}, avg_f1: {f1_a:.4f}")
        print(f"Set B → inside: {num_in_b}, avg_dist: {dist_b:.4f}, avg_f1: {f1_b:.4f}")
        print("---------------------------")

    # Step 1: more inside
    if num_in_a > num_in_b:
        overall_winner = "A"
        if verbose: print("Step 1 → A wins (more inside points)")
    elif num_in_b > num_in_a:
        overall_winner = "B"
        if verbose: print("Step 1 → B wins (more inside points)")
    # Step 2: smaller distance
    elif dist_a < dist_b - 1e-6:
        overall_winner = "A"
        if verbose: print("Step 2 → A wins (closer to rectangle)")
    elif dist_b < dist_a - 1e-6:
        overall_winner = "B"
        if verbose: print("Step 2 → B wins (closer to rectangle)")
    # Step 3: higher F1
    elif f1_a > f1_b + 1e-6:
        overall_winner = "A"
        if verbose: print("Step 3 → A wins (higher average F1)")
    elif f1_b > f1_a + 1e-6:
        overall_winner = "B"
        if verbose: print("Step 3 → B wins (higher average F1)")
    else:
        overall_winner = "tie"
        if verbose: print("Step 3 → tie")

    # === PER-CLASS COMPARISON ===
    all_classes = sorted(set(details_a.keys()) | set(details_b.keys()))
    per_class_winners = {}

    if verbose:
        print("\n=== Per-Class Comparison ===")

    for cls in all_classes:
        a = details_a.get(cls, {"precision": 0, "recall": 0, "f1": 0, "dist_to_zone": np.inf, "inside_zone": False})
        b = details_b.get(cls, {"precision": 0, "recall": 0, "f1": 0, "dist_to_zone": np.inf, "inside_zone": False})

        if verbose:
            print(f"[{cls}]")
            print(f"  A → prec: {a['precision']:.3f}, rec: {a['recall']:.3f}, f1: {a['f1']:.3f}, dist: {a['dist_to_zone']:.4f}, inside: {a['inside_zone']}")
            print(f"  B → prec: {b['precision']:.3f}, rec: {b['recall']:.3f}, f1: {b['f1']:.3f}, dist: {b['dist_to_zone']:.4f}, inside: {b['inside_zone']}")
            print("---------------------------")

    if return_details:
        return {
            "overall_winner": overall_winner,
            "summary": {
                "A": {"num_inside": num_in_a, "avg_dist": dist_a, "avg_f1": f1_a},
                "B": {"num_inside": num_in_b, "avg_dist": dist_b, "avg_f1": f1_b},
            },
            "per_class": {
                "A": details_a,
                "B": details_b,
                "winners": per_class_winners,
            }
        }

    return overall_winner


def compare_best_point_sets(best_points_a, best_points_b, prec_thresh=0.9, rec_thresh=0.8, verbose=True):
    """
    Compare two sets of best_points (outputs from find_best_pr_points).

    Prints internal decision steps if verbose=True.

    Logic:
      1. Prefer results with more points inside the "good zone".
      2. If equal, prefer smaller average distance to the rectangle.
      3. If still equal, prefer higher average F1.
    """

    def distance_to_rectangle(prec, rec, prec_thresh, rec_thresh):
        # distance to the rectangle (0 if inside)
        dx = max(0.0, prec_thresh - prec)
        dy = max(0.0, rec_thresh - rec)
        return np.sqrt(dx**2 + dy**2)

    def analyze_set(best_points):
        precisions = np.array([p["precision"] for p in best_points.values()])
        recalls = np.array([p["recall"] for p in best_points.values()])
        f1s = np.array([p["f1"] for p in best_points.values()])

        inside_mask = (precisions >= prec_thresh) & (recalls >= rec_thresh)
        num_inside = np.sum(inside_mask)
        avg_f1 = np.mean(f1s)
        dists = np.array([
            distance_to_rectangle(p, r, prec_thresh, rec_thresh)
            for p, r in zip(precisions, recalls)
        ])
        avg_dist = np.mean(dists)
        return num_inside, avg_f1, avg_dist

    num_in_a, f1_a, dist_a = analyze_set(best_points_a)
    num_in_b, f1_b, dist_b = analyze_set(best_points_b)

    if verbose:
        print("=== Comparison Summary ===")
        print(f"Thresholds: precision ≥ {prec_thresh}, recall ≥ {rec_thresh}\n")
        print(f"Set A → inside: {num_in_a}, avg_dist: {dist_a:.4f}, avg_f1: {f1_a:.4f}")
        print(f"Set B → inside: {num_in_b}, avg_dist: {dist_b:.4f}, avg_f1: {f1_b:.4f}")
        print("---------------------------")

    # Step 1: more points inside
    if num_in_a > num_in_b:
        if verbose: print("Step 1 result → A wins (more inside points)")
        return "A"
    elif num_in_b > num_in_a:
        if verbose: print("Step 1 result → B wins (more inside points)")
        return "B"

    # Step 2: smaller average distance
    if dist_a < dist_b - 1e-6:
        if verbose: print("Step 2 result → A wins (closer to rectangle)")
        return "A"
    elif dist_b < dist_a - 1e-6:
        if verbose: print("Step 2 result → B wins (closer to rectangle)")
        return "B"

    # Step 3: higher F1
    if f1_a > f1_b + 1e-6:
        if verbose: print("Step 3 result → A wins (higher average F1)")
        return "A"
    elif f1_b > f1_a + 1e-6:
        if verbose: print("Step 3 result → B wins (higher average F1)")
        return "B"

    if verbose: print("Step 3 result → tie (no meaningful difference)")
    return "tie"



model_number = "4"
base_output_path = f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/metrics/train{model_number}"
with open(os.path.join(base_output_path, "operating_points.json"), "r") as f:
    best_points = json.load(f)
with open(os.path.join(base_output_path, "pr_results.json"), 'r') as f:
        pr_results = json.load(f)

model_number = "8"
base_output_path = f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/metrics/train{model_number}"
with open(os.path.join(base_output_path, "operating_points.json"), "r") as f:
    second_points = json.load(f)

#plot_pr_curves(pr_results, best_points=best_points, second_points=second_points, base_output_path="/user/christoph.wald/u15287/insect_pest_detection/evaluate")

#print(compare_best_point_sets(best_points, second_points))
compare_best_point_sets_per_class(best_points, second_points)