import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/user/christoph.wald/u15287/insect_pest_detection/3_2_image_processing_evaluation/data/filtering_tests_collected_results.csv")
species = ['BRAIIM', 'LIRIBO', 'TRIAVA']
colors = {'BRAIIM': 'blue', 'LIRIBO': 'green', 'TRIAVA': 'red'}

for sp in species:
    # Filter rows where either _p or _FP is not NaN
    mask = df[f'{sp}_p'].notna() | df[f'{sp}_FP'].notna()
    df_filtered = df[mask]
    x = df_filtered['Unnamed: 0']
    
    fig, ax1 = plt.subplots(figsize=(10,5))
    
    # Left y-axis: _p values
    ax1.plot(x, df_filtered[f'{sp}_p'], marker='o', color=colors[sp], label=f'{sp} precision')
    ax1.set_ylabel('Precision', color=colors[sp])
    ax1.tick_params(axis='y', labelcolor=colors[sp])
    ax1.set_xticklabels(x, rotation=45, ha='right')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Right y-axis: _FP values
    ax2 = ax1.twinx()
    ax2.plot(x, df_filtered[f'{sp}_FP'], marker='x', linestyle='--', color=colors[sp], label=f'{sp} FP')
    ax2.set_ylabel('False positives', color=colors[sp])
    ax2.tick_params(axis='y', labelcolor=colors[sp])
    
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    plt.title(f'{sp} - thresholds')
    plt.tight_layout()
    plt.savefig(f"filter_tests_{sp}.jpg")