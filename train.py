import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(X, y, learning_rate, n_iterations):
    """Train the model. Returns the weights, the bias and the loss after every step."""
    X = np.array(X)                         
    y = np.array(y)

    n_samples = X.shape[0]
    n_features = X.shape[1]

    weights = np.zeros(n_features)         
    bias = 0.0
    loss_history = []

    for step in range(n_iterations):
        predictions = X.dot(weights) + bias          
        errors = predictions - y                    

        loss = np.mean(errors ** 2)                 
        loss_history.append(loss)

        weight_gradient = X.T.dot(errors) / n_samples   
        bias_gradient = np.mean(errors)

        weights = weights - learning_rate * weight_gradient
        bias = bias - learning_rate * bias_gradient

    return weights, bias, loss_history


def predict(X, weights, bias):
    """Use trained weights to predict values for new data."""
    return np.array(X).dot(weights) + bias


def plot_loss(loss_history, title):
    """Loss versus iteration - it should go down and then flatten."""
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (MSE)")
    plt.title(title)
    plt.yscale("log")                       
    plt.show()


def compare_learning_rates(X, y, learning_rates, n_iterations):
    """Train with several learning rates and draw all loss curves on one plot."""
    plt.figure(figsize=(8, 5))
    for rate in learning_rates:
        weights, bias, loss_history = gradient_descent(X, y, rate, n_iterations)
        plt.plot(loss_history, label="learning rate = " + str(rate))
        print("Learning rate", rate, "-> final loss:", round(loss_history[-1], 4))
    plt.xlabel("Iteration")
    plt.ylabel("Loss (MSE)")
    plt.yscale("log")
    plt.title("Effect of the learning rate")
    plt.legend()
    plt.show()
