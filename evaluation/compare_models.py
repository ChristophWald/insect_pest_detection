import os
import json
import numpy as np

'''
compare two sets of operating_points (outputs from find_best_pr_points given by eval_model.py).
'''

def compare_best_point_sets_per_class(best_points_a, best_points_b, prec_thresh=0.9, rec_thresh=0.8, verbose=True, return_details=True):
    """
   Logic:
      1. Prefer results with more points inside the target zone.
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


model_number = "14"
base_output_path = f"/user/christoph.wald/u15287/insect_pest_detection/3_3_self_training_evaluation/metrics/train{model_number}"
with open(os.path.join(base_output_path, "operating_points.json"), "r") as f:
    best_points = json.load(f)

model_number = "6"
base_output_path = f"/user/christoph.wald/u15287/insect_pest_detection/training/metrics/train{model_number}"
with open(os.path.join(base_output_path, "operating_points.json"), "r") as f:
    second_points = json.load(f)

compare_best_point_sets_per_class(best_points, second_points)