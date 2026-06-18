import re
import matplotlib.pyplot as plt
import os

log_file = "ppo_v2.log"
output_path = "/home/medhasree_2121cs05/.gemini/antigravity/brain/436ee0c8-acc9-4fcc-8d60-1daab4df538e/reward_profile.png"

iterations = []
mean_step_reward = []
launch_rate = []
win_rate = []
ship_margin = []

with open(log_file, "r") as f:
    text = f.read()

# Split into iteration blocks
blocks = re.split(r"ITERATION (\d+)", text)

for i in range(1, len(blocks), 2):
    iteration = int(blocks[i])
    block = blocks[i+1]
    
    m_reward = re.search(r"Mean Step Reward:\s+([-\+\.\d]+)", block)
    m_launch = re.search(r"Fleet Launch Rate:\s+([-\+\.\d]+)%", block)
    m_win = re.search(r"Epoch Win Rate:\s+([-\+\.\d]+)%", block)
    m_margin = re.search(r"Net Ship Margin:\s+([-\+\.\d]+)", block)
    
    if m_reward and m_launch and m_win and m_margin:
        iterations.append(iteration)
        mean_step_reward.append(float(m_reward.group(1)))
        launch_rate.append(float(m_launch.group(1)))
        win_rate.append(float(m_win.group(1)))
        ship_margin.append(float(m_margin.group(1)))

fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axs[0].plot(iterations, mean_step_reward, color='blue')
axs[0].set_title('Mean Step Reward')
axs[0].set_ylabel('Reward')
axs[0].grid(True, alpha=0.3)

axs[1].plot(iterations, launch_rate, color='red')
axs[1].set_title('Fleet Launch Rate (%)')
axs[1].set_ylabel('%')
axs[1].grid(True, alpha=0.3)

axs[2].plot(iterations, win_rate, color='green')
axs[2].set_title('Epoch Win Rate (%)')
axs[2].set_ylabel('%')
axs[2].grid(True, alpha=0.3)

axs[3].plot(iterations, ship_margin, color='purple')
axs[3].set_title('Net Ship Margin')
axs[3].set_xlabel('Iteration')
axs[3].set_ylabel('Ships')
axs[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
