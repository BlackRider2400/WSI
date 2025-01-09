import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.8
discount_factor = 0.95
epsilon = 1.0
epsilon_decay_rate = 0.9995
num_independent_runs = 25
num_episodes = 1000
max_steps = 200


def q_learning(env, reward_system):
    state_size = env.observation_space.n
    action_size = env.action_space.n

    averaged_rewards = np.zeros(num_episodes)

    for run in range(num_independent_runs):

        qtable = np.zeros((state_size, action_size))
        local_epsilon = epsilon

        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            truncated = False


            for i in range(max_steps):
                if np.random.rand() < local_epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(qtable[state, :])

                next_state, reward, done, truncated, _ = env.step(action)

                reward = reward_system(state, action, next_state, reward, done)

                qtable[state, action] += learning_rate * (
                        reward
                        + discount_factor * np.max(qtable[next_state, :])
                        - qtable[state, action]
                )

                state = next_state
                averaged_rewards[episode] += reward

                if truncated or done:
                    state, _ = env.reset()

            local_epsilon = max(local_epsilon * epsilon_decay_rate, 0.01)

    averaged_rewards /= num_independent_runs

    return averaged_rewards

def base_reward_system(state, action, next_state, reward, done):
    return reward


def custom_reward_system_1(state, action, next_state, reward, done):
    if state == next_state:
        return -0.05
    if done and reward == 1:
        return 1.0
    return reward + 0.01


def custom_reward_system_2(state, action, next_state, reward, done):
    if done and reward == 0:
        return -0.5
    if done and reward == 1:
        return 1.0
    return reward + 0.01


if __name__ == "__main__":
    # env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human')
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False)

    print("Calculating base non slippery.")
    base_rewards = q_learning(env, base_reward_system)
    print("Calculating custom 1 non slippery.")
    custom_rewards_1 = q_learning(env, custom_reward_system_1)
    print("Calculating custom 2 non slippery.")
    custom_rewards_2 = q_learning(env, custom_reward_system_2)

    plt.figure()
    plt.plot(range(num_episodes), base_rewards, label="Base Reward System", color='r')
    plt.plot(range(num_episodes), custom_rewards_1, label="Custom Reward System 1", color='b')
    # plt.plot(range(num_episodes), custom_rewards_2, label="Custom Reward System 2", color='g')
    plt.title("Average Rewards per Episode (Non-Slippery FrozenLake)")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig("frozenlake_non_slippery_1.png")
    plt.show()

    plt.figure()
    plt.plot(range(num_episodes), base_rewards, label="Base Reward System", color='r')
    # plt.plot(range(num_episodes), custom_rewards_1, label="Custom Reward System 1", color='b')
    plt.plot(range(num_episodes), custom_rewards_2, label="Custom Reward System 2", color='g')
    plt.title("Average Rewards per Episode (Non-Slippery FrozenLake)")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig("frozenlake_non_slippery_2.png")
    plt.show()

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True)
    num_episodes = 10000

    print("Calculating base slippery.")
    base_rewards = q_learning(env, base_reward_system)
    print("Calculating custom 1 slippery.")
    custom_rewards_1 = q_learning(env, custom_reward_system_1)
    print("Calculating custom 2 slippery.")
    custom_rewards_2 = q_learning(env, custom_reward_system_2)

    plt.figure()
    plt.plot(range(num_episodes), base_rewards, label="Base Reward System", color='r')
    plt.plot(range(num_episodes), custom_rewards_1, label="Custom Reward System 1", color='b')
    #plt.plot(range(num_episodes), custom_rewards_2, label="Custom Reward System 2", color='g')
    plt.title("Average Rewards per Episode (Slippery FrozenLake)")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig("frozenlake_slippery_1.png")
    plt.show()

    plt.figure()
    plt.plot(range(num_episodes), base_rewards, label="Base Reward System", color='r')
    #plt.plot(range(num_episodes), custom_rewards_1, label="Custom Reward System 1", color='b')
    plt.plot(range(num_episodes), custom_rewards_2, label="Custom Reward System 2", color='g')
    plt.title("Average Rewards per Episode (Slippery FrozenLake)")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig("frozenlake_slippery_2.png")
    plt.show()
