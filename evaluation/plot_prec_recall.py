import pandas as pd
import matplotlib.pyplot as plt

'''
simple plots for visualizing losses
'''
for i in range(1,17):
    # Load the CSV file
    file_path = f'/user/christoph.wald/u15287/insect_pest_detection/2_3_supervised_training/runs/detect/train{i}/results.csv'
    
    df = pd.read_csv(file_path)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Precision and Recall vs Epoch
    axs[0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision (B)', marker='o',ms= 2)
    axs[0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall (B)', marker='o',ms= 2)
    axs[0].plot(df['epoch'], df['metrics/mAP50(B)'], label='metrics/mAP50(B)', color='red', marker='o',ms= 2)
    axs[0].axhline(y=0.9, color='blue', linestyle=':', label='Precision Threshold')
    axs[0].axhline(y=0.8, color='orange', linestyle=':', label = "Recall Threshold")
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Score')
    axs[0].set_title('Precision and Recall vs Epoch')
    axs[0].legend()
    axs[0].grid(True)

    # Right plot: Training and Validation losses vs Epoch
    # Training losses
    axs[1].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', marker='o',ms= 2)
    axs[1].plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', marker='o',ms= 2)
    #axs[1].plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss', marker='o')

    # Validation losses (dashed lines)
    axs[1].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linestyle='--', marker='o',ms= 2)
    axs[1].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', linestyle='--', marker='o',ms= 2)
    #axs[1].plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss', linestyle='--', marker='o')

    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Training and Validation Losses vs Epoch')
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout and save the figure
    plt.tight_layout()
    #save_path = f'/user/christoph.wald/u15287/insect_pest_detection/3_1_supervised_training_evaluation/runs/detect/prec_recall_train{i}.png'
    save_path = f"/user/christoph.wald/u15287/insect_pest_detection/2_3_supervised_training/runs/detect/prec_recall_train{i}.png"
    plt.savefig(save_path)
    plt.close()
