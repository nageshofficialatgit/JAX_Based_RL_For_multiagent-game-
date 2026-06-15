import re
import matplotlib.pyplot as plt
import os
import argparse

def plot_training_log(log_path="training.log", output_path="training_metrics.png"):
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    # Regex to capture metrics from tqdm output
    # Example: Epoch 2: 2612it [19:57,  2.76it/s, Loss=6.145, V_Loss=0.989, Grad_Norm=4.523, SPS=612]
    pattern = re.compile(r"Loss=([\d.]+), V_Loss=([\d.]+), Grad_Norm=([\d.]+), SPS=(\d+)")

    steps = []
    losses = []
    v_losses = []
    grad_norms = []
    sps_values = []

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                loss, v_loss, grad_norm, sps = map(float, match.groups())
                steps.append(len(steps) * 10)  # Logged every 10 steps
                losses.append(loss)
                v_losses.append(v_loss)
                grad_norms.append(grad_norm)
                sps_values.append(sps)

    if not steps:
        print("No metrics found in the log file.")
        return

    print(f"Parsed {len(steps)} data points. Generating plots...")

    # Create a 2x2 grid of plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Orbit Wars Training Metrics', fontsize=16, fontweight='bold')

    # Plot Total Loss
    axs[0, 0].plot(steps, losses, color='tab:red', alpha=0.8, linewidth=2)
    axs[0, 0].set_title('Total Loss', fontsize=12)
    axs[0, 0].set_xlabel('Steps')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    # Plot Value Loss
    axs[0, 1].plot(steps, v_losses, color='tab:orange', alpha=0.8, linewidth=2)
    axs[0, 1].set_title('Value Loss', fontsize=12)
    axs[0, 1].set_xlabel('Steps')
    axs[0, 1].set_ylabel('V_Loss')
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)

    # Plot Gradient Norm
    axs[1, 0].plot(steps, grad_norms, color='tab:purple', alpha=0.8, linewidth=2)
    axs[1, 0].set_title('Gradient Norm', fontsize=12)
    axs[1, 0].set_xlabel('Steps')
    axs[1, 0].set_ylabel('Grad Norm')
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)

    # Plot SPS
    axs[1, 1].plot(steps, sps_values, color='tab:green', alpha=0.8, linewidth=2)
    axs[1, 1].set_title('States Per Second (SPS)', fontsize=12)
    axs[1, 1].set_xlabel('Steps')
    axs[1, 1].set_ylabel('SPS')
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved successfully to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize training metrics from log.')
    parser.add_argument('--log', type=str, default='training.log', help='Path to training log file')
    parser.add_argument('--out', type=str, default='training_metrics.png', help='Output image path')
    args = parser.parse_args()
    
    plot_training_log(args.log, args.out)
