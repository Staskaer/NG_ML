# ÊµÑéÔ¤²âº¯Êý

from sigmoid import sigmoid


def predict(theta, x):
    probability = sigmoid(x*theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]
