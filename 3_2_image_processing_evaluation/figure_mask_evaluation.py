import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

df = pd.read_csv("/user/christoph.wald/u15287/insect_pest_detection/3_2_image_processing_evaluation/mask_tests_on_labeled_data/revised_mask_testing_collected_results.csv")


# To avoid confusion with repeated mask labels, add index numbers
df["label"] = df.index + 1 # numerical x-axis for plotting
mask_labels = df["mask"] # store mask names separately for tick labels
mask_labels = ["Fig. 9b", "Fig. 9b", "Fig. 9c",  "Fig. 9f" ]
#"Fig. 9d", "Fig. 9e",
df = df[~df["label"].isin([4, 5])]
df["label"] = range(1, len(df) + 1)

# Plot all in one graph
plt.figure(figsize=(12, 8))


# Add background colors
plt.axvspan(0.5, 1.5, facecolor='lightyellow', alpha=0.5)
plt.axvspan(1.5, 4.5, facecolor='moccasin', alpha=0.5)

yellow_patch = mpatches.Patch(color='lightyellow', alpha=0.5, label='mask transformation')
orange_patch = mpatches.Patch(color='moccasin', alpha=0.5, label='+ extra shift')


# BRAIIM
plt.plot(df["label"], df["FP BRAIIM"], marker="o", linestyle="-", color="red", label="BRAIIM FP")
plt.plot(df["label"], df["TP BRAIIM"], marker="o", linestyle=":", color="red", label="BRAIIM TP")


# LIRIBO
plt.plot(df["label"], df["FP LIRIBO"], marker="o", linestyle="-", color="blue", label="LIRIBO FP")
plt.plot(df["label"], df["TP LIRIBO"], marker="o", linestyle=":", color="blue", label="LIRIBO TP")


# TRIAVA
plt.plot(df["label"], df["FP TRIAVA"], marker="o", linestyle="-", color="green", label="TRIAVA FP")
plt.plot(df["label"], df["TP TRIAVA"], marker="o", linestyle=":", color="green", label="TRIAVA TP")


# Replace x-axis ticks with mask names
plt.xticks(df["label"], mask_labels, rotation=30, ha="right")


plt.xlabel("Mask")
plt.ylabel("Count")
plt.grid(visible = True)
plt.title("Evaluating different masks")
plt.legend()
plt.legend(handles=[
    yellow_patch, orange_patch, 
    *plt.gca().get_legend_handles_labels()[0]  # keep existing FP/TP lines
], loc='best')
plt.tight_layout()
plt.savefig("figure_mask_evaluation.jpg")
