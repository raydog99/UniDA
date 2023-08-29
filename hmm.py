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

	def generate_seq(self, seq_length):
        X = np.zeros((seq_length, self.x_size))
        Z = np.zeros(seq_length)
        Z_pre = np.random.choice(self.n_state, 1, p=self.start_prob)
        X[0] = self.generate_x(Z_pre)
        Z[0] = Z_pre

        for i in range(seq_length):
            if i == 0: continue
            # P(Zn+1)=P(Zn+1|Zn)P(Zn)
            Z_next = np.random.choice(self.n_state, 1, p=self.transmat_prob[Z_pre,:][0])
            Z_pre = Z_next
            # P(Xn+1|Zn+1)
            X[i] = self.generate_x(Z_pre)
            Z[i] = Z_pre

        return X,Z

	def update():
		pass

	def train():
		pass

	