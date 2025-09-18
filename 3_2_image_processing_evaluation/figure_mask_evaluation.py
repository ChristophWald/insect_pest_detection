import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

df = pd.read_csv("/user/christoph.wald/u15287/insect_pest_detection/2_4_image_processing/data/mask_testing_collected_results.csv")


# To avoid confusion with repeated mask labels, add index numbers
df["label"] = df.index + 1 # numerical x-axis for plotting
mask_labels = df["mask"] # store mask names separately for tick labels


# Plot all in one graph
plt.figure(figsize=(12, 8))


# Add background colors
plt.axvspan(0.5, 2.5, facecolor='lightyellow', alpha=0.5)
plt.axvspan(2.5, 6.5, facecolor='moccasin', alpha=0.5)
plt.axvspan(6.5, 7.5, facecolor='lightcoral', alpha=0.5)

yellow_patch = mpatches.Patch(color='lightyellow', alpha=0.5, label='without mask transformation')
orange_patch = mpatches.Patch(color='moccasin', alpha=0.5, label='mask transformation')
red_patch = mpatches.Patch(color='lightcoral', alpha=0.5, label='+ extra shift')


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


plt.xlabel("Mask Type")
plt.ylabel("Count")
plt.grid(visible = True)
plt.title("Evaluating different masks")
plt.legend()
plt.legend(handles=[
    yellow_patch, orange_patch, red_patch,
    *plt.gca().get_legend_handles_labels()[0]  # keep existing FP/TP lines
], loc='best')
plt.tight_layout()
plt.savefig("figure_mask_evaluation.jpg")
