import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def plot_smoothed_rewards(file_paths, num_envs, hyperparams, title, algorithm, smoothing_window=10):
    plt.figure(figsize=(6, 6))  # Adjust the figure size to be more square

    for file_path, hyperparam in zip(file_paths, hyperparams):
        # Load the CSV file
        data = pd.read_csv(file_path, skiprows=1, header=0)

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

        # Apply smoothing
        results['smoothed_reward'] = results['average_reward'].rolling(window=smoothing_window).mean()

        # Plot the smoothed rewards
        plt.plot(results['time'], results['smoothed_reward'], label=f'{hyperparam}')

    # Adding labels and title
    plt.xlim(0, 12_000)
    plt.xlabel('Time (s)')
    plt.ylabel('Average Reward')
    plt.title(title)
    plt.legend(title='Hyperparameter Value')
    plt.grid(True)

    output_folder = "grid_search_plots/full_toolset/"

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Construct the file name
    file_name = f"{algorithm}_{title}.png"
    file_path = os.path.join(output_folder, file_name)

    # Save the plot
    plt.savefig(file_path)
    plt.close()


title = "Value Loss Coefficient"

# Example usage
file_paths = [
    "logs/baselines/june/new_grid_search/ppo/n_steps=256_map_w=24_clip_range=0.2_batch_size=128_"
    "n_epochs=10_v_l_coef=0.5_e_coef=0.01_lr=0.001_eps=1e-05_gamma=0.95_max_grad_norm=0.5_lambda="
    "0.95_seed=1_2024-05-31_23-01-13/vec_monitor_log.csv.monitor.csv",  # baseline run
    "logs/baselines/june/new_grid_search/ppo/v_l_coef/n_steps=256_map_w=24_clip_range=0.2_batch_size=128_n_epochs=10_v_l_coef=1.0_e_coef=0.01_lr=0.001_eps=1e-05_gamma=0.95_max_grad_norm=0.5_lambda=0.95_seed=1_2024-05-31_23-10-40/vec_monitor_log.csv.monitor.csv"
    ]
hyperparams = [
    '0.5',  # Corresponding hyperparameter for baseline
    '1.0'
    # Add more hyperparameters here
]

# Call the function
plot_smoothed_rewards(file_paths, num_envs=4, hyperparams=hyperparams, title=title,
                      algorithm="PPO", smoothing_window=10)
