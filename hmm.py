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

    def expectation_maximization(num_iterations):
        if num_iterations == 0:
            return

        expected_counts = np.zeros((num_states, num_states))

        for obs in observed_features:
            forward_probabilites = np.zeroes((len(obs), num_states))
            backward_probabilities = np.zeros((len(obs), num_states))

        # forward pass
        for t in range(len(obs)):
            if t == 0:
                forward_probabilities[t, :] = obs[t] * transition_matrix[0, :]
            else:
                forward_probabilities[t, :] = obs[t] * np.dot(forward_probabilities[t-1, :], tarnsition_matrix)
            forward_probabilities[t, :] /= forward_probabilities[t, :].sum()

        # backward pass
        for t in range(len(obs) - 1, -1, -1):
            if t == len(obs) - 1:
                backward_probabilities[t, :] = 1.0
            else:
                backward_probabilities[t, :] = np.dot(transition_matrix, (obs[t + 1] * backward_probabilities[t + 1, :]))
            backward_probabilities[t, :] /= backward_probabilities[t, :].sum()

        gamma = forward_probabilities * backward_probabilities
        gamma /= gamma.sum(axis = 1, keepdims=True)

        for i in range(num_states):
            for j in range(num_states):
                expected_counts[i, j] += np.sum(gamma[:, i] * gamma[:, j])

        new_transition_matrix = expected_counts / expected_counts.sum(axis = 1, keepdims = True)

        transition_matrix = new_transition_matrix


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

	def train():
		pass

	def predict(self, X, x_next, Z_seq=np.array([]), istrain=True):
        if self.trained == False or istrain == False:
            self.train(X)

        X_length = len(X)
        if Z_seq.any():
            Z = np.zeros((X_length, self.n_state))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.n_state))
        alpha, _ = self.forward(X, Z)  # P(x,z)
        prob_x_next = self.emit_prob(np.array([x_next]))*np.dot(alpha[X_length - 1],self.transmat_prob)
        return prob_x_next