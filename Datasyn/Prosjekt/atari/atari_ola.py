from __future__ import print_function, division
from builtins import range
import copy
import gym
import os
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from scipy.misc import imresize

ROOF_EXP = 500000
FLOOR_EXP = 50000
UPDATE_FREQ = 10000
SIZE_OF_IM = 84
ACTION_S = 4

class conv_2D_screen:
  def __init__(self):
    with tf.variable_scope("image_conversion"):
      self.stateIn = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
      self.output = tf.image.rgb_to_grayscale(self.stateIn)
      self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
      self.output = tf.image.resize_images(
        self.output,
        [SIZE_OF_IM, SIZE_OF_IM],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      self.output = tf.squeeze(self.output)

  def reconstruct(self, state, a=None):
    a = a or tf.get_default_session()
    return a.run(self.output, { self.stateIn: state })


def next_s(state, new_fr):
  return np.append(state[:,:,1:], np.expand_dims(new_fr, 2), axis=2)



class RerunExp:
  def __init__(self, amount=ROOF_EXP, ceil=SIZE_OF_IM, span=SIZE_OF_IM,
               length_mem=4, size_of_lot=32):
    self.amount = amount
    self.ceil = ceil
    self.span = span
    self.length_mem = length_mem
    self.size_of_lot = size_of_lot
    self.count = 0
    self.current = 0
    self.actions = np.empty(self.amount, dtype=np.int32)
    self.rewards = np.empty(self.amount, dtype=np.float32)
    self.frames = np.empty((self.amount, self.ceil, self.span), dtype=np.uint8)
    self.terminal_flags = np.empty(self.amount, dtype=np.bool)
    self.states = np.empty((self.size_of_lot, self.length_mem,
                            self.ceil, self.span), dtype=np.uint8)
    self.new_states = np.empty((self.size_of_lot, self.length_mem,
                                self.ceil, self.span), dtype=np.uint8)
    self.indices = np.empty(self.size_of_lot, dtype=np.int32)

  def gain_exp(self, move, frame, reward, terminal):

    if frame.shape != (self.ceil, self.span):
      raise ValueError('Wrong Dimension of frame!')
    self.actions[self.current] = move
    self.frames[self.current, ...] = frame
    self.rewards[self.current] = reward
    self.terminal_flags[self.current] = terminal
    self.count = max(self.count, self.current+1)
    self.current = (self.current + 1) % self.amount

  def state_now(self, idx):
    if self.count is 0:
      raise ValueError("Buffer is empty!")
    if idx < self.length_mem - 1:
      raise ValueError("Index to low (min 3)")
    return self.frames[idx-self.length_mem+1:idx+1, ...]

  def legal_idx(self):
    for i in range(self.size_of_lot):
      while True:
        idx = random.randint(self.length_mem, self.count - 1)
        if idx < self.length_mem:
          continue
        if idx >= self.current and idx - self.length_mem <= self.current:
          continue
        if self.terminal_flags[idx - self.length_mem:idx].any():
          continue
        break
      self.indices[i] = idx

  def mini_clust(self):

    if self.count < self.length_mem:
      raise ValueError('Not enough memory')

    self.legal_idx()

    for i, idx in enumerate(self.indices):
      self.states[i] = self.state_now(idx - 1)
      self.new_states[i] = self.state_now(idx)

    return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], self.rewards[self.indices], np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.terminal_flags[self.indices]


class Deep_Q_Net:
  def __init__(self, ACTION_S, size_conv, size_hidden, path):

    self.ACTION_S = ACTION_S
    self.path = path

    with tf.variable_scope(path):
      self.X = tf.placeholder(tf.float32, shape=(None, SIZE_OF_IM, SIZE_OF_IM, 4), name='X')
      self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
      self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
      out = self.X / 255.0
      for num_output_filters, filtersz, poolsz in size_conv:
        out = tf.contrib.layers.conv2d(
          out,
          num_output_filters,
          filtersz,
          poolsz,
          activation_fn=tf.nn.relu
        )
      out = tf.contrib.layers.flatten(out)
      for M in size_hidden:
        out = tf.contrib.layers.fully_connected(out, M)
      self.predict_op = tf.contrib.layers.fully_connected(out, ACTION_S)
      selected_action_values = tf.reduce_sum(
        self.predict_op * tf.one_hot(self.actions, ACTION_S),
        reduction_indices=[1]
      )
      cost = tf.reduce_mean(tf.losses.huber_loss(self.G, selected_action_values))
      self.train_op = tf.train.AdamOptimizer(1e-5).minimize(cost)
      self.cost = cost

  def mirror_network(self, other):
    cop2 = [t for t in tf.trainable_variables() if t.name.startswith(self.path)]
    cop2 = sorted(cop2, key=lambda v: v.name)
    copFrom = [t for t in tf.trainable_variables() if t.name.startswith(other.path)]
    copFrom = sorted(copFrom, key=lambda v: v.name)
    list = []

    for m, n in zip(cop2, copFrom):
      listemp = m.assign(n)
      list.append(listemp)
    self.session.run(list)


  def store_weights(self):
    parameters = [t for t in tf.trainable_variables() if t.name.startswith(self.path)]
    parameters = self.session.run(parameters)
    np.savez('weights_we_want2save.npz', *parameters)


  def get_weight_val(self):
    parameters = [t for t in tf.trainable_variables() if t.name.startswith(self.path)]
    npz = np.get_weight_val('tf_dqn_weights.npz')
    list = []
    for p, (_, v) in zip(parameters, npz.iteritems()):
      ops.append(p.assign(v))
    self.session.run(list)


  def determine_term(self, term):
    self.session = term

  def call(self, states):
    return self.session.run(self.predict_op, feed_dict={self.X: states})

  def renew(self, states, actions, targets):
    reval, _ = self.session.run(
      [self.cost, self.train_op],
      feed_dict={
        self.X: states,
        self.G: targets,
        self.actions: actions
      }
    )
    return reval

  def possible_action_epsGreed(self, x, epsilon):
    if np.random.random() < epsilon:
      return np.random.choice(self.ACTION_S)
    else:
      return np.argmax(self.call([x])[0])


def master(model, tmod, exp_buffer, disc_fac, size_of_lot):
  states, actions, rewards, next_states, dones = exp_buffer.mini_clust()

  next_Qs = tmod.call(next_states)
  next_Q = np.amax(next_Qs, axis=1)
  targets = rewards + np.invert(dones).astype(np.float32) * disc_fac * next_Q
  loss = model.renew(states, actions, targets)
  return loss


def play_one(
  env,
  sess,
  steps_in_total,
  exp_buffer,
  model,
  tmod,
  conv_im,
  disc_fac,
  size_of_lot,
  epsilon,
  delta_epsilon,
  epsilon_min):

  start_time = datetime.now()
  inspection = env.reset()
  inspection_small = conv_im.reconstruct(inspection, sess)
  state = np.stack([inspection_small] * 4, axis=2)
  loss = None
  time_in_training = 0
  epoch_steps = 0
  episode_reward = 0

  done = False
  while (done == False):
    if steps_in_total % UPDATE_FREQ == 0:
      tmod.mirror_network(model)

    move = model.possible_action_epsGreed(state, epsilon)
    inspection, reward, done, _ = env.step(move)
    inspection_small = conv_im.reconstruct(inspection, sess)
    next_state = next_s(state, inspection_small)
    episode_reward += reward
    exp_buffer.gain_exp(move, inspection_small, reward, done)
    start_time_2 = datetime.now()
    loss = master(model, tmod, exp_buffer, disc_fac, size_of_lot)
    dt = datetime.now() - start_time_2
    time_in_training += dt.total_seconds()
    epoch_steps += 1
    state = next_state
    steps_in_total += 1
    epsilon = max(epsilon - delta_epsilon, epsilon_min)

  return steps_in_total, episode_reward, (datetime.now() - start_time), epoch_steps, time_in_training/epoch_steps, epsilon


def get_average_retuns(that_sweet_reuturns):
  n = len(that_sweet_reuturns)
  list_avg = np.zeros(n)
  for j in range(n):
    start = max(0, j - 99)
    list_avg[j] = float(that_sweet_reuturns[start:(i+1)].sum()) / (j - start + 1)
  return list_avg

if __name__ == '__main__':

  size_conv = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
  size_hidden = [512]
  disc_fac = 0.99
  size_of_bat = 32
  how_much_to_train = 3500
  steps_in_total = 0
  exp_buffer = RerunExp()
  episode_rewards = np.zeros(how_much_to_train)

  epsilon = 1.0
  epsilon_min = 0.1
  delta_epsilon = (epsilon - epsilon_min) / 500000



  env = gym.envs.make("Breakout-v0")

  model = Deep_Q_Net(
    ACTION_S=ACTION_S,
    size_conv=size_conv,
    size_hidden=size_hidden,
    path="model")
  tmod = Deep_Q_Net(
    ACTION_S=ACTION_S,
    size_conv=size_conv,
    size_hidden=size_hidden,
    path="tmod"
  )
  conv_im = conv_2D_screen()

  with tf.Session() as sess:
    model.determine_term(sess)
    tmod.determine_term(sess)
    sess.run(tf.global_variables_initializer())


    print("Experience in buffer... this takes some time")
    inspection = env.reset()

    for i in range(FLOOR_EXP):

        action = np.random.choice(ACTION_S)
        inspection, reward, done, _ = env.step(action)
        inspection_small = conv_im.reconstruct(inspection, sess)
        exp_buffer.gain_exp(action, inspection_small, reward, done)

        if done:
            inspection = env.reset()


    start_time = datetime.now()
    for i in range(how_much_to_train):

      steps_in_total, episode_reward, duration, epoch_steps, time_per_step, epsilon = play_one(
        env,
        sess,
        steps_in_total,
        exp_buffer,
        model,
        tmod,
        conv_im,
        disc_fac,
        size_of_bat,
        epsilon,
        delta_epsilon,
        epsilon_min,
      )
      episode_rewards[i] = episode_reward

      avg_reward = episode_rewards[max(0, i - 100):i + 1].mean()

      print("Epoch num:", i,
        "Time of epoch:", duration,
        "Number of steps:", epoch_steps,
        "Reward:", episode_reward,
        "Training time per step:", "%.3f" % time_per_step,
        "Avg Reward (Last 100):", "%.3f" % avg_reward,
        "Epsilon:", "%.3f" % epsilon
      )
      sys.stdout.flush()
    print("Total duration:", datetime.now() - start_time)

    model.store_weights()

    avg_of_rewards_for_print = get_average_retuns(episode_rewards)
    plt.plot(episode_rewards, label='orig')
    plt.plot(avg_of_rewards_for_print, label='avgReturns')
    plt.legend()
plt.show()
