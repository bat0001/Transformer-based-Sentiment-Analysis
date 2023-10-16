import numpy as np

class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        self.embedding_matrix = np.random.rand(vocab_size, embedding_dim)
        self.grad_embedding_matrix = np.zeros_like(self.embedding_matrix)
    
    def forward(self, x):
        return self.embedding_matrix[x]

    def backward(self, grad_output, x):
        self.zero_grad()
        np.add.at(self.grad_embedding_matrix, x, grad_output)
        return self.grad_embedding_matrix

    def update_weights(self, learning_rate):
        """ Met à jour la matrice d'embedding en utilisant les gradients. """
        self.embedding_matrix -= learning_rate * self.grad_embedding_matrix
    
    def zero_grad(self):
        """ Remet à zéro les gradients. """
        self.grad_embedding_matrix.fill(0)

    def vectorize_sequence(self, sequence):
        """ Convertit une séquence d'ID de tokens en une séquence de vecteurs d'embedding. """
        return self.embedding_matrix[np.array(sequence)]

    def vectorize_data(self, data):
        """ Convertit des données composées de séquences d'ID de tokens en données composées de séquences d'embeddings. """
        return [self.vectorize_sequence(sequence) for sequence in data]

    def update_embeddings(self, learning_rate):
        self.embedding_matrix -= learning_rate * self.gradient_matrix
