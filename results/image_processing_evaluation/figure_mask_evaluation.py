import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import matplotlib.lines as mlines

'''
script for Figure 13
'''

df = pd.read_csv("/user/christoph.wald/u15287/insect_pest_detection/results/image_processing_evaluation/mask_tests_on_labeled_data/revised_mask_testing_collected_results.csv")


# To avoid confusion with repeated mask labels, add index numbers
df["label"] = df.index + 1 # numerical x-axis for plotting
mask_labels = df["mask"] # store mask names separately for tick labels
mask_labels = ["Fig. 7b single alignment", "Fig. 7b", "Fig. 7c",  "Fig. 7d" ]
#"Fig. 9d", "Fig. 9e",
df = df[~df["label"].isin([4, 5])]
df["label"] = range(1, len(df) + 1)

# Plot all in one graph
plt.figure(figsize=(12, 6))



# BRAIIM
plt.plot(df["label"], df["FP BRAIIM"], marker="o", linestyle=":", color="red", label="BRAIIM FP")
plt.plot(df["label"], df["TP BRAIIM"], marker="o", linestyle="-", color="red", label="BRAIIM TP")


# LIRIBO
plt.plot(df["label"], df["FP LIRIBO"], marker="o", linestyle=":", color="blue", label="LIRIBO FP")
plt.plot(df["label"], df["TP LIRIBO"], marker="o", linestyle="-", color="blue", label="LIRIBO TP")


# TRIAVA
plt.plot(df["label"], df["FP TRIAVA"], marker="o", linestyle=":", color="green", label="TRIAVA FP")
plt.plot(df["label"], df["TP TRIAVA"], marker="o", linestyle="-", color="green", label="TRIAVA TP")


# Replace x-axis ticks with mask names
plt.xticks(df["label"], mask_labels, rotation=30, ha="right")

# --- species (color) handles ---
species_handles = [
    mpatches.Patch(color='red', label='Fungus gnats'),
    mpatches.Patch(color='blue', label='Leaf miner flies'),
    mpatches.Patch(color='green', label='Whiteflies')
]

# --- TP/FP (linestyle) handles ---
fp_line = mlines.Line2D([], [], color='black', linestyle=':',  label='FP')
tp_line = mlines.Line2D([], [], color='black', linestyle='-', label='TP')

# --- combine into a single legend ---
all_handles = species_handles + [fp_line, tp_line]

plt.legend(handles=all_handles, loc='upper right', title="Legend", title_fontsize = 12, fontsize = 12, bbox_to_anchor=(0.9, 1.0))


plt.xlabel("Mask", fontsize = 12)
plt.ylabel("Count", fontsize = 12)
plt.grid(visible = True)
plt.title("Evaluating different masks", fontsize = 12)
plt.tight_layout()
plt.savefig("figure_mask_evaluation.jpg")
