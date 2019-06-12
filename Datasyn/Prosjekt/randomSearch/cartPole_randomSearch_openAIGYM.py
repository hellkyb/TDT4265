from __future__ import print_function, division
from builtins import range

import gym
import numpy as np
from bins import plot
import matplotlib.pyplot as plt

def calculate_dot_product(s, w):
    if(s.dot(w) > 0):
        return 1
    else:
        return 0

def train(environment, parameters):

    obs = environment.reset()
    done = False
    iterations = 0

    while(done == False and t < 10000):

        iterations += 1
        move = calculate_dot_product(obs, parameters)
        obs, reward, done, info = environment.step(move)
        if(done):
            break
    return iterations

def train_T_episodes(env, T, parameters, to_screen):
    ep_length = np.empty(T)

    for i in range(T):
        ep_length[i] = train(env, parameters)

    if(to_screen):
        print("Average Length: ", ep_length.mean())

    return(ep_length.mean())

def random_search_alg(env):
    run_avg = []
    ep_length = []
    leading = 0
    parameters = None

    for t in range(100):
        new_paramters = np.random.random(4)*2 - 1
        average_length = train_T_episodes(env, 100, new_paramters, False)

        ep_length.append(average_length)

        if(average_length > leading):
            parameters = new_paramters
            leading = average_length
    return ep_length, parameters

if __name__ == '__main__':
    to_screen = False
    env = gym.make('CartPole-v0')
    ep_lengths, parameters = random_search_alg(env)

    plt.plot(ep_lengths)
    plt.show()
    to_screen = True

    plot(average_length)

    print("LAST RUN \n")
    train_T_episodes(env, 100, parameters, to_screen)
