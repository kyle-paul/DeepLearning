import numpy as np

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def cross_entropy_loss(probs, targets):
    num_samples = targets.shape[0]
    correct_logprobs = -np.log(probs[range(num_samples), targets])
    loss = np.sum(correct_logprobs) / num_samples
    return loss

def compute_gradients(logits, probs, targets):
    num_samples = targets.shape[0]
    dscores = probs
    dscores[range(num_samples), targets] -= 1
    dscores /= num_samples
    
    dW = np.dot(logits.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    
    return dW, db
