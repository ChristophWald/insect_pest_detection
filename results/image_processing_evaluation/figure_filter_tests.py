import pandas as pd
import matplotlib.pyplot as plt

'''
script for figure 14
'''


df = pd.read_csv("//user/christoph.wald/u15287/insect_pest_detection/results/image_processing_evaluation/revised_filtering_tests_collected_results.csv")
species = ['BRAIIM', 'LIRIBO', 'TRIAVA']
new_labels = ['set1 ...', '+ contour size','+ overlaps', '+ box size', '+ box ratio', 'set2']  
df['Unnamed: 0'] = new_labels
# Colors correspond to curve type
curve_colors = {'p': 'blue', 'TP': 'green', 'FP': 'red'}
species_map = {'BRAIIM': 'Fungus gnats', 'LIRIBO': 'Leaf miner flies', 'TRIAVA': 'Whiteflies'}
species_colors = {'BRAIIM': 'red', 'LIRIBO': 'blue', 'TRIAVA': 'green'}

for sp in species:
    # Filter rows where any of the three values is not NaN
    mask = df[f'{sp}_p'].notna() | df[f'{sp}_TP'].notna() | df[f'{sp}_FP'].notna()
    df_filtered = df[mask]
    x = df_filtered['Unnamed: 0']

    fig, ax1 = plt.subplots(figsize=(4,5))

    # Left y-axis: Precision
    ax1.plot(x, df_filtered[f'{sp}_p'], marker='o', linestyle='--', markersize=3, linewidth=1, color='black', label='Precision')

    ax1.set_ylabel('Precision', fontsize = 12)
    ax1.tick_params(axis='y')
    ax1.set_xticklabels(x, rotation=45, ha='right', fontsize = 12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax1.set_ylim(0.7, 1.0)


    # Right y-axis: Absolute counts (TP and FP)
    ax2 = ax1.twinx()
    ax2.plot(x, df_filtered[f'{sp}_TP'], marker='o', linestyle='-', color=species_colors[sp], label='TP')
    ax2.plot(x, df_filtered[f'{sp}_FP'], marker='o', linestyle=':', color=species_colors[sp], label='FP')

    ax2.set_ylabel('Counts (TP / FP)', fontsize = 12)
    ax2.tick_params(axis='y')

    ax2.set_ylim(0, 4000)
    ax1.set_xlabel("filter", fontsize = 12)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize = 12)

    plt.title(f'{species_map[sp]}', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"filter_tests_{sp}.jpg")
    plt.close()
