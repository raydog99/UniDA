import numpy as np

class HiddenMarkovModel:
	def __init__(self, n_state, x_size):
		self.n_state = n_state
		self.x_size = x_size
		self.start_prob = np_ones(n_state) * 1.0 / n_state
		self.transmat_prob = np.ones((n_state, n_state) * 1.0 / n_state)

	@classmethod
	def initialize():
		pass

	def update():
		pass

	def train():
		pass