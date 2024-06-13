import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_multiple_timesteps_vs_rewards(file_paths, labels, title):
    plt.figure(figsize=(6, 6))

    for file_path, label in zip(file_paths, labels):
        # Load the CSV file
        data = pd.read_csv(file_path)

        # Extract the relevant columns
        timesteps = data['time/total_timesteps']

        # Apply smoothing
        data['smoothed_reward'] = data['rollout/ep_rew_mean'].rolling(window=1).mean()
        rewards = data['smoothed_reward']
        # Plot the data
        plt.plot(timesteps, rewards, label=label)

    # Adding labels and title with increased font size
    plt.xlabel('Total Timesteps', fontsize=16)
    plt.ylabel('Episode Reward Mean', fontsize=16)
    plt.title(title, fontsize=16)
    plt.xlim(0, 2_000_000)

    # Adjust tick label font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    plt.legend()
    plt.grid(True)

    output_folder = "plots/grid_search"

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Construct the file name
    file_name = "A2C_" + f"{title}.png"
    file_path = os.path.join(output_folder, file_name)

    # Save the plot
    plt.savefig(file_path)
    plt.close()


#
# def plot_smoothed_rewards(file_paths, num_envs, hyperparams, title, algorithm, smoothing_window=10):
#     plt.figure(figsize=(6, 6))  # Adjust the figure size to be more square
#
#     for file_path, hyperparam in zip(file_paths, hyperparams):
#         # Load the CSV file
#         data = pd.read_csv(file_path, skiprows=1, header=0)
#
#         # Initialize lists to store averaged rewards and corresponding times
#         average_rewards = []
#         times = []
#
#         # Iterate over the data in chunks of 'num_envs' rows
#         for i in range(0, len(data), num_envs):
#             chunk = data.iloc[i:i + num_envs]
#             avg_reward = chunk['r'].mean()
#             time = chunk['t'].iloc[0]  # Assuming all 't' values in the chunk are the same
#             average_rewards.append(avg_reward)
#             times.append(time)
#
#         # Create a DataFrame with the results
#         results = pd.DataFrame({'time': times, 'average_reward': average_rewards})
#
#         # Apply smoothing
#         results['smoothed_reward'] = results['average_reward'].rolling(window=smoothing_window).mean()
#
#         # Plot the smoothed rewards
#         plt.plot(results['time'], results['smoothed_reward'], label=f'{hyperparam}')
#
#     # Adding labels and title
#     plt.xlim(0, 10_000)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Average Reward')
#     plt.title(title)
#     plt.legend(title='Hyperparameter Value')
#     plt.grid(True)
#
#     output_folder = "power_puzzle_plots"
#
#     # Ensure the output folder exists
#     os.makedirs(output_folder, exist_ok=True)
#
#     # Construct the file name
#     file_name = f"{algorithm}_{title}.png"
#     file_path = os.path.join(output_folder, file_name)
#
#     # Save the plot
#     plt.savefig(file_path)
#     # plt.show()
#     plt.close()



if __name__ == "__main__":
    title = "Value Loss Coefficient"

    # Example usage
    file_paths = [
        "logs/baselines/june/new_grid_search/a2c/n_steps=20_map_w=24_gamma=0.95_v_l_coef=0.5_e_coef=0.01_max_grad_norm=0.5_lr=0.001_seed=1_eps=1e-05_lambda=0.95_2024-06-01_11-03-28/progress.csv"
        ,"logs/baselines/june/new_grid_search/a2c/v_l_coef/n_steps=20_map_w=24_gamma=0.95_v_l_coef=1.0_e_coef=0.01_max_grad_norm=0.5_lr=0.001_seed=1_eps=1e-05_lambda=0.95_2024-06-01_03-57-40/progress.csvg"

    ]
    labels = [
        '0.5',  # Corresponding label for baseline
        '1.0',
        # Add more hyperparameters here
    ]

    plot_multiple_timesteps_vs_rewards(file_paths, labels, title)
