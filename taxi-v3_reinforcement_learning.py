#
# Reinforcement learning with OpenAI Gym Taxi-v3
# https://gym.openai.com/envs/Taxi-v3/
#

import gym
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import statistics as stats


# Run manual episode with user inputs
def manual_episode():
    
    # Set the environment
    env.reset()
    env.render()

    # Run manual loop
    print('Press e to quit and r to reset.')
    print('Movements: 0 (S), 1 (N), 2 (E), 3 (W), 4 (PU), 5 (DO)')
    mov = ''
    while (True):
        mov = input('Move: ')
        if(mov == 'e'):
            break
        elif(mov == 'r'):
            env.reset()
            env.render()
        else:
            state, reward, done, info = env.step(int(mov))
            print('State:', state)
            env.render()
            if done:
                print('Reward:', reward)
                env.reset()
                env.render()


# Run training episodes without user input
def training_episodes(num_of_ep):
    print('Starting training...')
    start_time = time.time()

    # Q-learning parameters
    alpha = 0.9 # Learning rate
    gamma = 0.9 # Future reward discount factor
    num_of_episodes = num_of_ep # Number of training episodes
    num_of_steps = 50 # per each episode
    
    # Training parameters
    epsilon = 1 # Exploration rate
    max_epsilon = 1 # Exploration prob at start
    min_epsilon = 0.1 # Min exploration prob
    decay = 0.01 # Decay rate for exploration prob

    # Q tables for rewards
    q_table = np.zeros((500,6))

    # Table to log training rewards
    training_rewards = []
    training_steps = []

    # Run the training episodes
    for episode in range(num_of_episodes):
        
        print('Starting episode: {}/{}'.format(episode, num_of_episodes))

        # Set the environment
        state = env.reset()
        tot_reward = 0

        # Start movement
        for step in range(num_of_steps):
            start_step = random.uniform(0, 1)

            if start_step > epsilon:
                random_step = False
                action = np.argmax(q_table[state,:]) # Select best action
            else:
                random_step = True
                action = env.action_space.sample() # Take random action

            # Run action
            new_state, reward, done, info = env.step(action)
            # print('Step number: {}, current state: {}, action: {}, random step: {}, reward: {}'.format(
            #     step, state, action, random_step, reward)) # used for debugging
            
            # Update Q-table
            q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])
            tot_reward += reward
            
            # Update the state
            state = new_state

            if done:
                print('Total reward for episode {}: {}'.format(episode, tot_reward))
                training_rewards.append(tot_reward)
                training_steps.append(step)
                break
            
            # Reduce the random movement
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
    
    # Print training results
    print('Training ready.')
    training_time = time.time()-start_time
    print('Run time: {:2.2f} s.'.format(training_time))
    print('Completed training episodes: {}'.format(num_of_episodes))
    print('Successful episodes: {}'.format(len(training_rewards)))
    print('Success rate: {:2.2f} %'.format(
        (len(training_rewards)/num_of_episodes) * 100))
    print('Average steps: {:2.2f}, average reward: {:2.2f}'.format(
        stats.mean(training_steps), stats.mean(training_rewards)))
    time.sleep(1)
    return q_table, training_rewards, training_steps


# Run testing episodes for the Q-table
def test_episodes(num_of_ep, q_table):
    print('Starting testing...')

    # Q-learning parameters
    num_of_episodes = num_of_ep # Number of training episodes
    num_of_steps = 50 # per each episode

    # Table to log training rewards
    testing_rewards = []
    testing_steps = []

    # Run the training episodes
    for episode in range(num_of_episodes):
        
        print('Starting episode: {}/{}'.format(episode, num_of_episodes))

        # Set the environment
        state = env.reset()
        tot_reward = 0

        # Start movement
        for step in range(num_of_steps):
            
            # Select best action
            action = np.argmax(q_table[state,:])

            # Run action
            new_state, reward, done, info = env.step(action)
            # print('Step number: {}, current state: {}, action: {}, reward: {}'.format(
            #     step, state, action, reward)) # used for debugging

            # Update the state
            tot_reward += reward
            state = new_state

            if done:
                print('Total reward for episode {}: {}'.format(episode, tot_reward))
                testing_rewards.append(tot_reward)
                testing_steps.append(step)
                break
    
    # Print training results
    print('Testing ready.')
    print('Completed testing episodes: {}'.format(num_of_episodes))
    print('Successful episodes: {}'.format(len(testing_rewards)))
    print('Success rate: {:2.2f} %'.format(
        (len(testing_rewards)/num_of_episodes) * 100))
    print('Average steps: {:2.2f}, average reward: {:2.2f}'.format(
        stats.mean(testing_steps), stats.mean(testing_rewards)))
    time.sleep(1)
    return testing_rewards, testing_steps


def main():
    # Define environment
    env = gym.make("Taxi-v3")
    print('Starting OpenAI Taxi-v3...')
    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    
    # User parameters
    man_input = False
    
    if(man_input):
        manual_episode()
    else:
        q_table, training_rewards, training_steps = training_episodes(50000)
        testing_rewards, testing_steps = test_episodes(100, q_table)
        
        # Plot results
        plt.figure()

        # Training
        plt.subplot(221)
        plt.title('Training rewards')
        plt.plot(training_rewards)
        plt.grid(True)
        plt.subplot(222)
        plt.title('Training steps')
        plt.plot(training_steps)
        plt.grid(True)

        # Testing
        plt.subplot(223)
        plt.title('Testing rewards')
        plt.plot(testing_rewards)
        plt.grid(True)
        plt.subplot(224)
        plt.title('Testing steps')
        plt.plot(testing_steps)
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()