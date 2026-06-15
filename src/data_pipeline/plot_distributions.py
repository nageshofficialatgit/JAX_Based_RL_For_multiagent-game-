import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load data
    df = pd.read_csv('/home/medhasree_2121cs05/2201cs50_nagesh/server_deploy/parquet_db_real/agent_behavior_profiles.csv')
    
    # Filter out agents with very few games to remove noise (>= 50 games)
    df = df[df['total_games'] >= 50]
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#1e1e2e')
    
    metrics = [
        ('snipe_vulnerability_rate', 'Snipe Vulnerability Rate (%)', 100),
        ('avg_capture_roi', 'Average Capture ROI', 1),
        ('coordinated_strike_rate', 'Coordinated Strike Rate (%)', 100),
        ('attacks_per_turn', 'Attacks Per Turn', 1)
    ]
    
    # Define colors
    main_color = '#00ffcc'
    mean_color = '#ff3366'
    
    for i, (col, title, mult) in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        
        # Multiply by 100 for percentages
        data = df[col] * mult
        mean_val = data.mean()
        
        # Plot histogram
        ax.hist(data, bins=20, color=main_color, edgecolor='black', alpha=0.7)
        ax.set_facecolor('#2d2d3d')
        
        # Add mean line
        ax.axvline(mean_val, color=mean_color, linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        
        # Highlight top 5 players (by win rate)
        top5 = df.sort_values(by='win_rate', ascending=False).head(5)
        for _, row in top5.iterrows():
            val = row[col] * mult
            ax.plot(val, 0, marker='^', markersize=12, color='#ffd700', alpha=0.9) # Gold triangles at bottom
            
        ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Value', color='white', fontsize=12)
        ax.set_ylabel('Frequency (Number of Players)', color='white', fontsize=12)
        ax.tick_params(colors='white')
        
        # Add legend
        ax.plot([], [], marker='^', color='#ffd700', linestyle='None', markersize=10, label='Top 5 Grandmasters')
        legend = ax.legend(facecolor='#1e1e2e', edgecolor='none', labelcolor='white')
        
    plt.suptitle('Tactical Meta-Game Distributions (Min 50 Games)', color='white', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save directly to artifact directory
    out_path = '/home/medhasree_2121cs05/.gemini/antigravity/brain/436ee0c8-acc9-4fcc-8d60-1daab4df538e/tactical_distributions.png'
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    main()
