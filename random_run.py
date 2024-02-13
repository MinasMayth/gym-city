import gym
import gym_city


def make_env():
    env = gym.make("MicropolisEnv-v0")
    env.setMapSize(9, render_gui=False)
    return env


max_env_steps = 200


def run_random_actions(env, num_steps):
    for step in range(num_steps):
        # Take a random action from the action space
        action = env.action_space.sample()

        # Perform the action in the environment
        observation, reward, done, _ = env.step(action)

        print(reward)

        # Render the environment (if needed)
        env.render()

        time.sleep(1)

        # Check if the episode is done (e.g., the agent reached a terminal state)
        if done:
            print("Episode finished after {} timesteps".format(step + 1))
            break


def main():
    env = make_env()
    env.reset()
    # Set the number of steps to run the environment
    num_steps = 1000

    # Run the environment with random actions
    run_random_actions(env, num_steps)

    # Close the environment
    env.close()


if __name__ == "__main__":
    main()
