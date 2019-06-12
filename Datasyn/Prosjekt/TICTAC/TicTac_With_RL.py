
from __future__ import print_function, division
from builtins import range, input

import numpy as np
import matplotlib.pyplot as plt

#Global_varibles
ROW = 3
COL = 3

class Environment:
    def __init__(self):
        #Make a 3x3 board
        #Assign symbols
        #Assign winner and if game is finnished
        self.board = np.zeros((ROW, COL))
        self.o = 1 # Player O is 1
        self.x = -1 #Player x is -1
        self.winner = None
        self.game_over = False
        self.num_states = 3**(ROW*COL)

    def is_tile_empty(self, i, j):
        return self.board[i,j] == 0

    def reward(self, symbol):
        if not self.is_game_over():
            return 0
        else:
            if self.winner == symbol:
                return 1
            else:
                return 0

    def current_state(self):
        a,b,c = 0, 0, 0
        for i in range(ROW):
            for j in range(COL):
                if self.board[i,j] == 0:
                    c = 0
                elif self.board[i,j] == self.x:
                    c = 1
                elif self.board[i,j] == self.o:
                    c = 2
                b += (3**a)*c
                a += 1
        return b

    def is_game_over(self, force_recalculate=False):
        if(force_recalculate == False and self.game_over):
            return self.game_over
        for i in range(ROW):
            for player in (self.x, self.o):
                if self.board[i].sum() == player*3:
                    # Three in a row
                    self.winner = player
                    self.game_over = True
                    print("player ", player, " wins \n")
                    return True
        for j in range(COL):
            for player in (self.x, self.o):
                if self.board[:,j].sum() == player*3:
                    #Three in a column
                    self.winner = player
                    self.game_over = True
                    return True


        for player in (self.x, self.o):
            if self.board.trace() == player*3:
                #3 on the diagonal top left - bottom frame_height
                self.winner = player
                self.game_over = True
                return True
            elif np.fliplr(self.board).trace() == player*3:
                #Other diagonal
                self.winner = player
                self.game_over = True
                return True

        if np.all((self.board == 0) != True):
            #All tiles are full, but no winners
            self.winner = None
            self.game_over = True
            print("Draw \n")
            return True
        self.winner = None
        return False

    def draw(self):
        #Returns bool
        val = self.game_over and self.winner is None
        return val

    def print_board(self):
        for i in range(ROW):
          print("-------------")
          for j in range(COL):
            print("  ", end="")
            if self.board[i,j] == self.x:
              print("x ", end="")
            elif self.board[i,j] == self.o:
              print("o ", end="")
            else:
              print("  ", end="")
          print("")
        print("-------------")

class Agent:
    def __init__(self, epsilon=0.1, alpha=0.4):
        self.epsilon = epsilon
        self.alpha = alpha
        self.state_history = []
        self.interact = False

    def setV(self, value_function):
        #
        self.value_function = value_function

    def set_symbol(self, symbol):
        self.symbol = symbol

    def display_info(self, info):
        self.info = info

    def clear_history(self):
        self.state_history = []

    def make_move(self, board):
        best_state = None
        random_num = np.random.rand()
        if random_num < self.epsilon:
            #Random Action (Epsilon greedy)
            moves = []

            for i in range(ROW):
            #Iterate through rows
                for j in range(COL):
                #Iterate thorugh columns
                    if board.is_empty(i, j):
                        moves.append((i, j))
            index = np.random.choice(len(moves))
            next_move = moves[index]
        else:
        # If we are not choosing random_num
            next_move = None
            best_value = -1
            pos2val = {}

        for i in range(ROW):
            for j in range(COL):
                if board.is_empty(i, j):
                    board.board[i,j] = self.symbol
                    state = board

class Agent:
    def __init__(self, eps=0.1, alpha=0.6):
        self.eps = eps
        self.alpha = alpha
        self.give_info_pls =False
        self.state_history = []

    def set_val_func(self, V):
        self.V = V

    def choose_symbol(self, symbol):
        self.symbol = symbol

    def set_info_bool(self, v):
        self.give_info_pls = v

    def reset_history(self):
        self.state_history = []

    def do_move(self, env):
        random_num = np.random.rand()
        best_state = None

        #We do the epsilon greedy
        if random_num < self.eps:
            if self.give_info_pls:
                print("Random")
            possible_moves = []

            for i in range(ROW):
                for j in range(COL):
                    if env.is_tile_empty(i,j):
                        possible_moves.append((i,j))
            index = np.random.choice(len(possible_moves))
            next_move = possible_moves[index]
        else:
            pos2val = {}
            next_move = None
            best_value = -1
            for i in range(ROW):
                for j in range(COL):
                    if env.is_tile_empty(i,j):
                        env.board[i,j] = self.symbol
                        state = env.current_state()
                        env.board[i,j] = 0
                        pos2val[(i,j)] = self.V[state]
                        if self.V[state] > best_value:
                            best_value = self.V[state]
                            best_state = state
                            next_move = (i,j)
        if self.give_info_pls:
            print("Non-Random move")
            for i in range(ROW):
                print("-------------")
                for j in range(COL):
                    if env.is_tile_empty(i,j):
                        print(" %.2f|" % pos2val[(i,j)], end="")
                    else:
                        print("  ", end="")
                        if env.board[i,j] == env.x:
                            print("x  |", end="")
                        elif env.board[i,j] == env.o:
                            print("o  |", end="")
                        else:
                            print("  |", end="")
                print("")
            print("------------------")

        env.board[next_move[0], next_move[1]] = self.symbol

    def update_state_history(self, state):
        self.state_history.append(state)

    def update(self, env):
        reward = env.reward(self.symbol)
        target = reward
        for prev in reversed(self.state_history):
            value = self.V[prev] + self.alpha*(target - self.V[prev])
            self.V[prev] = value
            target = value

        self.reset_history()




class Human:
  def __init__(self):
    pass

  def set_symbol(self, sym):
    self.sym = sym

  def do_move(self, env):
    while True:
      # break if we make a legal move
      move = input("Enter coordinates i,j for your next move (i,j=0..2): ")
      i, j = move.split(',')
      i = int(i)
      j = int(j)
      if env.is_tile_empty(i, j):
        env.board[i,j] = self.sym
        break

  def update(self, env):
    pass

  def update_state_history(self, s):
    pass


def get_state_hash_and_winner(env, i=0, j=0):
  results = []

  for v in (0, env.x, env.o):
    env.board[i,j] = v # if empty board it should already be 0
    if j == 2:
      # j goes back to 0, increase i, unless i = 2, then we are done
      if i == 2:
        # the board is full, collect results and return
        state = env.current_state()
        ended = env.is_game_over(force_recalculate=True)
        winner = env.winner
        results.append((state, winner, ended))
      else:
        results += get_state_hash_and_winner(env, i + 1, 0)
    else:
      # increment j, i stays the same
      results += get_state_hash_and_winner(env, i, j + 1)

  return results

def initialV_x(env, state_winner_triples):
  # initialize state values as follows
  # if x wins, V(s) = 1
  # if x loses or draw, V(s) = 0
  # otherwise, V(s) = 0.5
  V = np.zeros(env.num_states)
  for state, winner, ended in state_winner_triples:
    if ended:
      if winner == env.x:
        v = 1
      else:
        v = 0
    else:
      v = 0.5
    V[state] = v
  return V


def initialV_o(env, state_winner_triples):
  # this is (almost) the opposite of initial V for player x
  # since everywhere where x wins (1), o loses (0)
  # but a draw is still 0 for o
  V = np.zeros(env.num_states)
  for state, winner, ended in state_winner_triples:
    if ended:
      if winner == env.o:
        v = 1
      else:
        v = 0
    else:
      v = 0.5
    V[state] = v
  return V


def play_game(p1, p2, env, draw=False):
  # loops until the game is over
  current_player = None
  while not env.is_game_over():
    # alternate between players
    # p1 always starts first
    if current_player == p1:
      current_player = p2
    else:
      current_player = p1

    # draw the board before the user who wants to see it makes a move
    if draw:
      if draw == 1 and current_player == p1:
        env.print_board()
      if draw == 2 and current_player == p2:
        env.print_board()


    current_player.do_move(env)

    # update state histories
    state = env.current_state()
    p1.update_state_history(state)
    p2.update_state_history(state)

  if draw:
    env.print_board()

  # do the value function update
  p1.update(env)
  p2.update(env)


if __name__ == '__main__':
  # train the agent
  p1 = Agent()
  p2 = Agent()

  # set initial V for p1 and p2
  env = Environment()
  state_winner_triples = get_state_hash_and_winner(env)


  Vx = initialV_x(env, state_winner_triples)
  p1.set_val_func(Vx)
  Vo = initialV_o(env, state_winner_triples)
  p2.set_val_func(Vo)

  # give each player their symbol
  p1.choose_symbol(env.x)
  p2.choose_symbol(env.o)

  T = 10000
  for t in range(T):
    if t % 200 == 0:
      print(t)
    play_game(p1, p2, Environment())

  # play human vs. agent
  # do you think the agent learned to play the game well?
  human = Human()
  human.set_symbol(env.o)
  while True:
    p1.set_info_bool(True)
    play_game(p1, human, Environment(), draw=2)
    # I made the agent player 1 because I wanted to see if it would
    # select the center as its starting move. If you want the agent
    # to go second you can switch the human and AI.
    answer = input("Play again? [Y/n]: ")
    if answer and answer.lower()[0] == 'n':
      break
