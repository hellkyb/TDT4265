from __future__ import print_function, division
from builtins import range
import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from bins import plot

class HiddenLayer:
  def __init__(self, A, B, f=tf.nn.tanh, bias=True):
    self.Weights = tf.Variable(tf.random_normal(shape=(A, B)))
    self.parameters = [self.Weights]
    self.bias = bias
    if bias:
      self.b = tf.Variable(np.zeros(B).astype(np.float32))
      self.parameters.append(self.b)
    self.f = f

  def forward(self, X):
    if self.bias:
      for_val = tf.matmul(X, self.Weights) + self.b
    else:
      for_val = tf.matmul(X, self.Weights)
    return self.f(for_val)


class DeepNet:
  def __init__(self, num_inputs, output_actions, hidden_layer_sizes, dicount_fac, maximum_exp=10000, min_exp=100, batch_size=32):
    self.output_actions = output_actions
    self.layers = []
    A = num_inputs
    for B in hidden_layer_sizes:
      layer = HiddenLayer(A, B)
      self.layers.append(layer)
      A = B
    layer = HiddenLayer(A, output_actions, lambda x: x)
    self.layers.append(layer)
    self.parameters = []
    for layer in self.layers:
      self.parameters += layer.parameters

    self.X = tf.placeholder(tf.float32, shape=(None, num_inputs), name='X')
    self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
    self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

    output = self.X
    for layer in self.layers:
      output = layer.forward(output)
    out_hat = output
    self.predict_op = out_hat

    chosen_action_val = tf.reduce_sum(
      out_hat * tf.one_hot(self.actions, output_actions),
      reduction_indices=[1]
    )

    cost = tf.reduce_sum(tf.square(self.G - chosen_action_val))

    self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)

    self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
    self.maximum_exp = maximum_exp
    self.min_exp = min_exp
    self.batch_size = batch_size
    self.dicount_fac = dicount_fac

  def det_round(self, huddle):
    self.huddle = huddle

  def image_inputN(self, refference):

    list = []
    my_parameters = self.parameters
    parameters_to_copy = refference.parameters
    for i, j in zip(my_parameters, parameters_to_copy):
      actual = self.huddle.run(j)
      listnum = i.assign(actual)
      list.append(listnum)

    self.huddle.run(list)

  def think(self, X):
    X = np.atleast_2d(X)
    return self.huddle.run(self.predict_op, feed_dict={self.X: X})

  def gain_experience(self, target_network):

    if len(self.experience['s']) < self.min_exp:
      return


    random_idx = np.random.choice(len(self.experience['s']), size=self.batch_size, replace=False)

    states = [self.experience['s'][i] for i in random_idx]
    actions = [self.experience['a'][i] for i in random_idx]
    rewards = [self.experience['r'][i] for i in random_idx]
    next_states = [self.experience['s2'][i] for i in random_idx]
    dones = [self.experience['done'][i] for i in random_idx]
    next_Q = np.max(target_network.think(next_states), axis=1)
    targets = [r + self.dicount_fac*next_q if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]


    self.huddle.run(
      self.train_op,
      feed_dict={
        self.X: states,
        self.G: targets,
        self.actions: actions
      }
    )

  def store_in_memory(self, s, a, r, s2, done):
    if len(self.experience['s']) >= self.maximum_exp:
      self.experience['s'].pop(0)
      self.experience['a'].pop(0)
      self.experience['r'].pop(0)
      self.experience['s2'].pop(0)
      self.experience['done'].pop(0)
    self.experience['s'].append(s)
    self.experience['a'].append(a)
    self.experience['r'].append(r)
    self.experience['s2'].append(s2)
    self.experience['done'].append(done)

  def choose_action_to_do(self, x, epsilon):
    if np.random.random() < epsilon:
      return np.random.choice(self.output_actions)
    else:
      X = np.atleast_2d(x)
      return np.argmax(self.think(X)[0])


def balance_cart_one_time(env, model, target, epsilon, dicount_fac, copy_period):
    done = False
    infoEnv = env.reset()

    totalreward = 0
    iteration = 0
    while( (done == False) and (iteration < 5000)):
        action = model.choose_action_to_do(infoEnv, epsilon)
        prev_infoEnv = infoEnv
        infoEnv, reward, done, info = env.step(action)

        totalreward += reward
        if done == True:
            reward = -350

    model.store_in_memory(prev_observation, action, reward, observation, done)
    model.gain_experience(target)

    iteration += 1

    if iteration % copy_period == 0:
      target.image_inputN(model)

  return totalreward


def main():
  env = gym.make('CartPole-v0').env
  dicount_fac = 0.99
  ref_session = 50

  num_inputs = len(env.observation_space.sample())
  output_actions = env.action_space.n
  sizes = [200,200]
  model = DeepNet(num_inputs, output_actions, sizes, dicount_fac)
  target = DeepNet(num_inputs, output_actions, sizes, dicount_fac)
  init = tf.global_variables_initializer()
  huddle = tf.InteractiveSession()
  huddle.run(init)
  model.det_round(huddle)
  target.det_round(huddle)

  N = 0
  n = 0
  totalrewards = np.empty(N)
  while True:


    epsilon = 1.0/np.sqrt(n+1)
    if n >= 100:
        epsilon = 0.1


    totalreward = balance_cart_one_time(env, model, target, epsilon, dicount_fac, ref_session)



    if totalreward > 10000:
        break
    totalrewards = np.append(totalrewards, totalreward)

    if totalrewards[max(0, n-20):(n+1)].mean() > 10000000:
        break
    n += 1
  plt.plot(totalrewards)
  plt.title("Rewards")
  plt.show()
  filename = os.path.basename(__file__).split('.')[0]
  monitor_dir = './' + filename + '_' + "LastRUNnnn"
  env = wrappers.Monitor(env, monitor_dir)
  plot(totalrewards)
  balance_cart_one_time(env, model, target, epsilon, dicount_fac, ref_session)
  print("Shitty CartPole is now less shitty")


if __name__ == '__main__':
  main()
