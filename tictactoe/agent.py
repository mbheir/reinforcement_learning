import numpy as np
import copy

from tictactoe import TicTacToe


class Agent:
    
	def __init__(self) -> None:
		# Values and rewards
		self.R, U = self.initStateValues()
		self.U = self.valueIteration(self.R,U)
		
	def get_optimal_action(self,state) -> tuple[dict,dict]:
		game = TicTacToe(self_init=True, state_str=state)
		Q_values,state_actions = game.get_Q_values_for_actions(self.R,self.U)
		Q_max = max(Q_values)
		Q_max_indx = Q_values.index(Q_max)

		optimal_action = state_actions[Q_max_indx]
		return optimal_action, Q_values
	
	def initStateValues(self,base_reward=1):
		C = ['.', 'x', 'o']
		S = set()
		U = {}
		R = {}
		for c9 in C:
			for c8 in C:
				for c7 in C:
					for c6 in C:
						for c5 in C:
							for c4 in C:
								for c3 in C:
									for c2 in C:
										for c1 in C:
											state = c9+c8+c7+c6+c5+c4+c3+c2+c1

											if state.count('x') != state.count('o')+1:
												continue  # check if valid state

											game = TicTacToe(
												self_init=True, state_str=state)
											is_termial_state, player_won, computer_won = game.is_terminal()

											if is_termial_state:
												if player_won:
													R[state] = 10
												elif computer_won:
													R[state] = -10
												else:
													R[state] = 0  # tie
											else:
												R[state] = base_reward
											U[state] = copy.deepcopy(R[state])
		return R, U

	def valueIteration(self,R,U):
		gamma = 0.9
		target = 0.1
		delta = target*(1 - gamma)/gamma + 0.01
		U = copy.deepcopy(U)
		U_ = copy.deepcopy(U)
		iterations = 0
		while delta > target*(1 - gamma)/gamma:
		# for i in range(40):
			delta = 0
			for s in U.keys():
			# Get possible next states
				game = TicTacToe(self_init=True,state_str=s)
				is_termial,_,_ = game.is_terminal()
				if is_termial: action_states = []
				else: action_states = game.get_possible_next_positions(player='o')

				maxsum = -np.inf
				if not action_states:
					maxsum = 0
				# print("start")
				for a in action_states:
					children = TicTacToe(self_init=True,state_str=a).get_possible_next_positions(player='x')
					if children != []:
						sum_ = 1/len(children) * sum([U[c] for c in children]) #p(s'|s,a) * sum(U[s'])
					else: 
						sum_ = 0
					maxsum = max(maxsum,sum_)
					# print(sum_)
				# print(f"maxsum: {maxsum}")

				#Bellman update
				U_[s] = R[s] + gamma * maxsum
				# print(U_[s])

				# Update delta
				if abs(U_[s] - U[s]) > delta: 
					delta = abs(U_[s] - U[s])
			U = copy.deepcopy(U_)
			iterations += 1
		print(f"Iterations: {iterations}")
		return U_


def ai_play_uniform(game,agent,should_print=True):
	while True:
		game.env_random_move()
		game.states.append(copy.deepcopy(game.state))
		# self.print_state()

		is_terminal, player_won, computer_won = game.is_terminal()
		if is_terminal:
			if player_won:
				game.rewards.append(10)
				if should_print:
					game.print_states()
				return True, game.state
			elif computer_won:
				game.rewards.append(-10)
				if should_print:
					game.print_states()
				return False, game.state
			else:
				game.rewards.append(0)
				if should_print:
					game.print_states()
				return False, game.state

		action_state,Q_value = agent.get_optimal_action(game.state_to_string_enc(game.state))
		print(Q_value)
		map = {'.': 0, 'o': 1, 'x': -1}
		game.state = np.array([map[s] for s in action_state]).reshape((3, 3))
		# game.actions.append(move)
		game.nState += 1
		game.rewards.append(1)

def play_against_ai(game,agent):
	while True:
		is_terminal, ai_won, player_won = game.is_terminal()
		if player_won:
			print(f"Congratz, you won")
		elif ai_won:
			print(f"Sorry, you lost")
		elif is_terminal:
			print(f"Draw")
			
		game.print_state(game.state)
		if is_terminal:
			return
		move = game.manual_move(piece="x")
		game.states.append(copy.deepcopy(game.state))


		action_state,Q_value = agent.get_optimal_action(game.state_to_string_enc(game.state))
		print(Q_value)

		map = {'.': 0, 'o': 1, 'x': -1}
		game.state = np.array([map[s] for s in action_state]).reshape((3, 3))
		# game.actions.append(move)
		game.nState += 1
		game.rewards.append(1)

def test_state(agent,state):
	print("\nTest state:")
	game = TicTacToe(self_init=True,state_str=state)
	game.print_state(game.state)
	action_state,Q_value = agent.get_optimal_action(state)
	print(Q_value)


if __name__=="__main__":
	agent = Agent()
	tictactoe = TicTacToe()
	# ai_play_uniform(game=tictactoe,agent=agent)
	play_against_ai(tictactoe,agent)

	# test_state(agent,".ox.x.ox.")
	# test_state(agent,"....x....")




