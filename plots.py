import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = \
    "logs/baselines/may/nuevo_grid_search/ppo/alpha=0.99_num_steps=256_map_width=24_clip_range=0.2_batch_size=128_n_epochs=10_value_loss_coef=0.5_entropy_coef=0.01_lr=0.001_eps=1e-05_gamma=0.95_max_grad_norm=0.5_lambda=0.95_seed=1_2024-05-31_17-24-24/vec_monitor_log.csv.monitor.csv"
data = pd.read_csv(file_path, skiprows=1, header=0)


# Number of environments
num_envs = 4

# Initialize lists to store averaged rewards and corresponding times
average_rewards = []
times = []

# Iterate over the data in chunks of 'num_envs' rows
for i in range(0, len(data), num_envs):
    chunk = data.iloc[i:i + num_envs]
    avg_reward = chunk['r'].mean()
    time = chunk['t'].iloc[0]  # Assuming all 't' values in the chunk are the same
    average_rewards.append(avg_reward)
    times.append(time)

# Create a DataFrame with the results
results = pd.DataFrame({'time': times, 'average_reward': average_rewards})

# Plot the average rewards over time
plt.figure(figsize=(10, 6))
plt.plot(results['time'], results['average_reward'], marker='o')
plt.xlabel('Time')
plt.ylabel('Average Reward')
plt.title('Average Reward over Time')
plt.grid(True)
plt.show()
