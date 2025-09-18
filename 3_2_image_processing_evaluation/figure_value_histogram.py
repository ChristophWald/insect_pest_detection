import ast
import matplotlib.pyplot as plt
import numpy as np

values_file = "/user/christoph.wald/u15287/insect_pest_detection/3_2_image_processing_evaluation/data/values.txt"

values = []

with open(values_file, "r") as f:
    for line in f:
        line = line.strip()
        if line:  # skip empty lines
            v = ast.literal_eval(line)
            values.extend(v)

print(np.percentile(values, 5))

plt.hist(values, bins=40, color='lightgreen', edgecolor='black')
plt.axvline(np.percentile(values, 5), color='green', linestyle='--', label='5th percentile')
plt.title(f"Triava color values")
plt.xlabel("Value")
plt.ylabel("Count")
plt.legend()
plt.savefig("value_histogram.jpg")