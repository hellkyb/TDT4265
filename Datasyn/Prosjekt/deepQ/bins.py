from __future__ import print_function, division
from builtins import range
import gym
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

def convert_state_to_num(features):
    # Features: List of integers
    # Returns integer representation of string

    val = int("".join(map(lambda feature: str(int(feature)), features)))

    return val


def convert_to_bin(number, bins):
    #Takes a number and an array of possible bins
    # Returns the bin the number belongs to
    return np.digitize(x=[number], bins=bins)[0]


class ConvertFeatures:
    def __init__(self):
        # Initiate physhical limits
        self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
        self.cart_velocity_bins = np.linspace(-2, 2, 9)
        self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
        self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9)

    def convert(self, obs):
        # Turn observation into integer
        cart_position, cart_velocity, pole_angle, pole_velocity = obs

        temp = convert_state_to_num([convert_to_bin(cart_position, self.cart_position_bins),
        convert_to_bin(cart_velocity, self.cart_velocity_bins),
        convert_to_bin(pole_angle, self.pole_angle_bins),
        convert_to_bin(pole_velocity, self.pole_velocity_bins)])

        return temp


class Model:
    def __init__(self, environment, conv_feature):
        # Initialize Q table and instances
        self.environment = environment
        self.conv_feature = conv_feature
        number_of_states = 10**environment.observation_space.shape[0]
        number_of_actions = environment.action_space.n
        self.Q = np.random.uniform(low=-1, high=1, size=(number_of_states, number_of_actions))

    def call(self, state):
        # State to integer for indexing of Q- table
        # Returns Q for all actions Q[idx] - 1D array
        idx = self.conv_feature.convert(state)
        return self.Q[idx]

    def update_Q(self, state, action, G):
        # G: target Return
        # Convert State into integer
        # Update Q by gradient descent

        idx = self.conv_feature.convert(state)
        self.Q[idx, action] += 1e-2*(G - self.Q[idx, action])

    def case_of_action(self, state, epsilon):
        #Epsilon Greedy
        if( np.random.random()< epsilon):
            # Pick a random action in from the action space
            temp = self.environment.action_space.sample()
            return temp
        else:
            # Follow the policy
            call_val = self.call(state)
            max_call_val = np.argmax(call_val)
            return max_call_val


def train(model, epsilon, discount_rate):

    experience = environment.reset()
    complete = False
    total_reward = 0
    iterations = 0

    while(complete == False and iterations < 10000):
        response = model.case_of_action(experience, epsilon)
        last_experience = experience
        experience, reward, complete, info = environment.step(response)

        total_reward += reward

        if(complete and iterations < 199):
            reward = -300

        G = reward + discount_rate*np.max(model.call(experience))
        model.update_Q(last_experience, response, G)
        iterations += 1
    return total_reward

def plot(total_rewards):
        # Function for plottinh the running average

    T = len(total_rewards)
    avg = np.empty(T)

    for i in range(T):
        avg[i] = total_rewards[max(0, i-100):(i+1)].mean()

    plt.plot(avg)
    plt.show()

if __name__ == '__main__':
    # In order to save a video, pass 'video' as sys argument in the terminal
    # i.e pyton3 bins.py video
    #
    # Main func:
    # Create the gym environment
    # Create ConvertFeatures Object
    # Sett the discount factor GAMMA
    # Sett iterations T
    # Dynamic epsilon
    # Update total_rewards



    environment = gym.make('CartPole-v0')
    conv_feut = ConvertFeatures()
    model = Model(environment, conv_feut)
    discount_factor = 0.8

    if 'video' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        video_directory = './' + filename + './' + "vid3"
        environment = wrappers.Monitor(environment, video_directory)

    T = 10000

    total_rewards = np.empty(T)


    #Training LOOP
    for i in range(T):
        epsilon = 1.0/np.sqrt(i+1)
        total_reward = train(model, epsilon, discount_factor)
        total_rewards[i] = total_reward

        # Print every 250 iteration

        if(i % 250 == 0):
            print("Epoch: ", i, "Total reward: ", total_reward, "Epsilon: ", epsilon)

    print("Average over last 100 epoch: ", total_rewards[-100:].mean())
    print("total steps: ", total_rewards.sum())

    plt.plot(total_rewards)
    plt.show()

    plot(total_rewards)
