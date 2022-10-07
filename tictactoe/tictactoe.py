import matplotlib.pyplot as plt
import numpy as np
import random
import copy


class TicTacToe:

    def __init__(self, self_init=False, state_str=0):
        # TODO: init rewards for self_initialized state
        if not self_init:
            self.state = np.array([[0]*3]*3)
        else:
            map = {'.': 0, 'o': 1, 'x': -1}
            self.state = np.array([map[s] for s in state_str]).reshape((3, 3))
        self.nTiles = 3
        self.nState = 0
        self.states = []
        self.rewards = []
        self.actions = []
        self.gamma = 0.9

    def place_tile(self, index, piece):
        self.state[index[0], index[1]] = piece

    def get_open_tiles(self):
        s = self.state
        open_tiles = []
        for m in range(s.shape[0]):
            for n in range(s.shape[1]):
                if s[m][n] == 0:
                    open_tiles.append((m, n))
        return open_tiles

    # returns string encodings of states
    def get_possible_next_positions(self, player='x'):
        map = {'.': 0, 'o': 1, 'x': -1}
        children = []
        open_tiles = self.get_open_tiles()
        if open_tiles == []:
            return []
        state = copy.deepcopy(self.state)
        for tile in open_tiles:
            self.place_tile(tile, map[player])
            children.append(self.state_to_string_enc(self.state))
            self.state = copy.deepcopy(state)
        return children

    def state_to_string_enc(self, state):
        map = {0: '.', 1: 'o', -1: 'x'}
        s = ''
        for c in state.flatten():
            s += map[c]
        return s

    def env_random_move(self):
        open_tiles = self.get_open_tiles()
        tile = random.choice(open_tiles)  # uniform distribution
        self.place_tile(tile, -1)

    def print_state(self, state):
        s = state
        map = {0: ' ', 1: 'o', -1: 'x'}
        # print(f'State_{len(self.states)}')
        print(' ------------------')
        for i in range(s.shape[0]):
            print(f'|', end='')
            for j in range(s.shape[1]):
                print(f'  {map[s[i][j]]:<3}|', end='')
            print('')
        print(' -----------------')

    def get_reward_to_go(self, n, rewards):
        reward_to_go = 0
        for i in range(len(rewards)-n):
            reward_to_go += self.gamma**i * rewards[n+i]
        return reward_to_go

    def print_states(self):
        for n, state in enumerate(self.states):
            self.print_state(state)
            reward_to_go = self.get_reward_to_go(n, self.rewards)
            print(f'reward-to-go={reward_to_go}')

    def manual_move(self,piece="x"):
        tile = input().split()
        move = (int(tile[0]), int(tile[1]))
        map = {'.': 0, 'o': 1, 'x': -1}
        self.place_tile(move,map[piece])

    def ai_move(self):
        open_tiles = self.get_open_tiles()
        tile = random.choice(open_tiles)  # uniform distribution
        return tile

    def is_terminal(self):  # ->is_terminal,player_won,computer_won
        for row in self.state:
            if sum(row) == 3:
                return True, True, False

        for i in range(self.state.shape[1]):
            if sum(self.state[:, i]) == 3:
                return True, True, False

        dia1, dia2 = 0, 0
        for i in range(self.nTiles):
            dia1 += self.state[0+i, 0+i]
            dia2 += self.state[0+i, 2-i]
        if dia1 == 3 or dia2 == 3:
            return True, True, False

        for row in self.state:
            if sum(row) == -3:
                return True, False, True
        for i in range(self.state.shape[1]):
            if sum(self.state[:, i]) == -3:
                return True, False, True
        if dia1 == -3 or dia2 == -3:
            return True, False, True

        if self.get_open_tiles() == []:
            return True, False, False

        # if none of the above, not a terminal state
        return False, False, False
    
    def get_Q_values_for_actions(self,R,V):
        Q_values = []
        state_actions = self.get_possible_next_positions('o')

        for state_action in state_actions:
            children = TicTacToe(self_init=True,state_str=state_action).get_possible_next_positions('x')
            P = 1/len(children)
            Q = R[self.state_to_string_enc(self.state)] + self.gamma * P * sum([V[c] for c in children])
            Q_values.append(Q)

        return Q_values,state_actions

    def play_game(self, print=False):
        while True:
            self.env_random_move()
            self.states.append(copy.deepcopy(self.state))
            # self.print_state()

            is_terminal, player_won, computer_won = self.is_terminal()
            if is_terminal:
                if player_won:
                    self.rewards.append(10)
                    if print:
                        self.print_states()
                    return True, self.state
                elif computer_won:
                    self.rewards.append(-10)
                    if print:
                        self.print_states()
                    return False, self.state
                else:
                    self.rewards.append(0)
                    if print:
                        self.print_states()
                    return False, self.state

            move = self.ai_move()
            self.place_tile(move, 1)
            self.actions.append(move)
            self.nState += 1
            self.rewards.append(1)

if __name__=="__main__":
    game = TicTacToe()
    game.play_game(print=True)
