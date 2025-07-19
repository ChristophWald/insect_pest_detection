import json

# Path to your file
json_file_path = "/user/christoph.wald/u15287/insect_pest_detection/results_testing.txt"  # ← Replace with your actual file path

# Initialize totals
TP_total = 0
FP_total = 0
FN_total = 0

# Read file line by line
with open(json_file_path, "r") as f:
    for line in f:
        entry = json.loads(line.strip())
        TP_total += entry.get("TP", 0)
        FP_total += entry.get("FP", 0)
        FN_total += entry.get("FN", 0)

# Compute overall metrics
precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0.0
recall = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0.0

# Print results
print(f"Total TP: {TP_total}, FP: {FP_total}, FN: {FN_total}")
print(f"Overall Precision: {precision:.4f}")
print(f"Overall Recall: {recall:.4f}")
