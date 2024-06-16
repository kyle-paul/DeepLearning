import numpy as np
from utils import softmax, cross_entropy_loss, compute_gradients



class Linear:
    def __init__(self, input_dim, output_dim, bias=True):
        limit = np.sqrt(6 / input_dim)
        self.weights = np.random.uniform(-limit, limit, (input_dim, output_dim))
        if bias:
            self.biases = np.zeros((1, output_dim))
    
    def forward(self, x):
        self.last_input = x
        return np.dot(x, self.weights) + self.biases
    
    def backward(self, dW, db, learning_rate=0.01):
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db
        

x = np.random.randn(1, 128)
y_true = np.array([2]) 

fc = Linear(128, 3)
logits = fc.forward(x)
probs = softmax(logits)
loss = cross_entropy_loss(probs, y_true)

# Compute gradients
dW, db = compute_gradients(fc.last_input, probs, y_true)

# Update weights and biases
fc.backward(dW, db)

print("Loss:", loss)
print("Updated Weights:", fc.weights)
print("Updated Biases:", fc.biases)
