import numpy as np
import pandas as pd
import math
import random

from typing import Tuple, Mapping

from sympy.abc import x, y
from sympy import diff
from sympy import cos, sin, exp, pi


def mse_grad(X: pd.Series, 
             Y: pd.Series, 
             w: np.array) -> np.array:
    
    """Calculate mean squared error gradient for dataset
        
       Args:
           X (pd.Series): features matrix with ONES column
           Y (pd.Series): target vector
           w (np.array): weights

       Return:
           np.array: gradients
    """
    
    y_hat = X @ w
    error = X.T @ y_hat - X.T @ Y
    grad = (2 / X.shape[0]) * error

    return grad


def func_grad(X: np.array, 
              Y: np.array, 
              w: np.array, 
              grad_f: Mapping) -> np.array:
    
    """Calculate mean squared error gradient for dataset
        
       Args:
           X (pd.Series): features matrix with ONES column
           Y (pd.Series): target vector
           w (np.array): weights
           grad_f (Mapping): gradient function

       Return:
           np.array: gradients
    """
    
    X_len = X.shape[0]
    
    if X_len == 1:
        X, Y = grad_f(X[0], Y[0])
        
    else:
        X, Y = grad_f(X, Y)
        X, Y = X.mean(), Y.mean()
    
    w = np.array(grad_f(w[0], w[1]))
    y_hat = X * w
    error = X * y_hat - X * Y
    grad = (2 / X_len) * error
    
    return grad


class GradientDescent():      
    results = {'weights': [],
               'grads': []}
    
    
    def __init__(self, 
                 X: np.array, 
                 Y: np.array,
                 grad_f: Mapping = None,
                 minim_f: Mapping = mse_grad,
                 epsilon: int = 2e-4,
                 seed: int = 2020):
        
        """Initialize GradientDescent
        
           Args:
               X (np.array): features matrix
               Y (np.array): target vector
               grad_f (Mapping, optional): gradient function. Defaults to None
               minim_f (Mapping, optional): minimization function. Defaults to mse_grad
               epsilon (int, optional): epsilon value. Defaults to 2e-4
               seed(int, optional): seed value. Defaults to 2020
        """
        
        self.X = X
        self.Y = Y
        self.grad_f = grad_f
        self.minim_f = minim_f
        self.epsilon = epsilon
        self.seed = seed
        
        self.X_ones = np.hstack(((np.ones((X.shape[0], 1))), X[:, None]))
        
        if seed is None:
            np.random.seed(0)
        
    
    def descent(self,  
                w0: tuple,
                descent: str,
                learn_r: float = 0.01,
                max_iters: int = 300,
                batch_size: int = None,
                threshold: int = 2e-4) -> dict:
        
        """Gradient descent calculation
        
           Args:
               w0 (tuple): start weights
               descent (string, optional): type of SGD
               learn_r (float, optional): learning rate. Defaults to 0.01
               max_iters (int, optional): number of max iterations. Defaults to 300
               batch_size (int, optional): batch size. Defaults to None
               threshold (int, optional): epsilon value. Defaults to 2e-4
                
           Return:
               Tuple [list, list]: weights, grads
        """
        
        self.clear_results(w0)
        
        w_old = w0
        iters = 1
        dw = np.array(2 * self.epsilon)
        
        while abs(dw).sum() > threshold and iters <= max_iters:
            w_new, dw = self.grad_condition(w_old, descent, learn_r, batch_size)
            
            w_old = w_new
            iters += 1
        
        # self.print_min(iters, w_old)
        
        return self.results
    
    
    def grad_condition(self, w_old, descent, learn_r, batch_size) -> Tuple:
        """Gradient type"""
        
        if descent == 'gradient':
            return  self.get_gradients(self.X_ones, self.X, self.Y, 
                                            w_old, descent, learn_r)

        elif descent == 'stochastic':
            return self.stochastic_descent(w_old, descent, learn_r)

        elif descent == 'minibatch':
            return self.minibatch_descent(w_old, descent, learn_r, batch_size)

        else:
            raise Exception('Incorrect type of descent')
    
    
    def stochastic_descent(self, w_old, descent, learn_r) -> Tuple:
        """Stochastic gradient descent algorithm"""
        
        i = np.random.randint(self.X_ones.shape[0])
        
        return self.get_gradients(self.X_ones[i, None], self.X[i, None], 
                                  self.Y[i, None], w_old, descent, learn_r)
    
    
    def minibatch_descent(self, w_old, descent, learn_r, batch_size) -> Tuple:
        """Stochastic mini-batch gradient descent algorithm"""
        
        batches_count = self.X.shape[0] // batch_size

        for i in range(batches_count):
            begin = i * batch_size
            end = (i + 1) * batch_size

            w_new, dw = self.get_gradients(self.X_ones[begin:end, :], self.X[begin:end], 
                                           self.Y[begin:end], w_old, descent, learn_r)
            
        return w_new, dw
    
    
    def get_gradients(self, X_ones, X, Y, w_old, descent, learn_r) -> Tuple:
        """Calculate gradients per each loop saving the result"""
        
        try:
            grad = self.minim_f(X_ones, Y, w_old)
            
        except:
            if descent == 'gradient':
                grad = self.grad_f(*w_old)
                
            else:
                grad = self.minim_f(X, Y, w_old, self.grad_f)
        
        dw = learn_r * grad
        
        w_new = w_old - dw

        self.results['weights'].append(list(w_new))
        self.results['grads'].append(list(grad))
        
        return w_new, dw
    
    
    def clear_results(self, w0):
        """Clear results"""
        self.results['weights'] = [w0]
        self.results['grads'] = [[0, 0]]
    
    
    def print_min(self, iters, w):
        """Print terminated minimum"""
        
        print('Got to the min:')
        print(f'iter: {iters}\nw0: {w[0]:.2f}\nw1: {w[1]:.2f}\n')


class Optimization:
    results = []

    def __init__(self, 
                 weights: list, 
                 grads: list, 
                 beta1: float = 0.9, 
                 beta2: float = 0.999,
                 epsilon: int = 2e-4):
        
        """Initialize Optimization
        
        Args:
               weights (list): gradient weights
               grads (list): gradients
               beta1 (float, optional): 1st coefficient. Defaults to 0.9
               beta2 (float, optional): 2nd coefficient. Defaults to 0.999
               epsilon (int, optional): epsilon value. Defaults to 2e-4
        """
        
        self.weights = np.array(weights)
        self.grads = np.array(grads)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
    
    def optimize(self, 
                 optimizer: str,
                 learn_r: float = 0.01) -> np.array:
        
        """Calculate optimization
           
           Args:
               optimizer (str): type of optimizer
               learn_r (float, optional): learning rate. Defaults to 0.01
           
           Return:
               np.array: updated weights
               
        """
        
        self.clear_results()
        
        if optimizer == 'adam':
            return self.adam_optimizer(learn_r)
            
        elif optimizer == 'momentum':
            return self.momentum_optimizer(learn_r)
            
        else:
            raise Exception('Incorrect type of optimization')
    
    
    def adam_optimizer(self, learn_r) -> np.array:
        """Adam optimization"""
        
        m = [0] * self.weights.shape[0]
        v = [0] * self.weights.shape[0]
        
        iters = 1
        
        for i, (weight, grad) in enumerate(zip(self.weights, self.grads)): 
            m[i] = self.beta1 * m[i] + (1 - self.beta1) * grad          
            v[i] = self.beta2 * v[i] + (1 - self.beta2) * grad ** 2

            m_corrected = m[i] / (1 - self.beta1 ** iters)
            v_corrected = v[i] / (1 - self.beta2 ** iters)

            new_weight = weight - learn_r * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
            iters += 1
        
            self.results.append(new_weight)
            
        # self.print_min(self.results[-1])
        
        return self.results
    

    def momentum_optimizer(self, learn_r) -> np.array:
        """Momentum optimization"""
        
        previous_updates = [0] * self.weights.shape[0]
        prevs = []
        mom_coeff = 1
        
        for weight, grad, prev_update in zip(self.weights, self.grads, previous_updates):
            delta = learn_r * grad - mom_coeff * prev_update
            new_weight = weight - delta
            
            prevs.append(delta)
            self.results.append(new_weight)
            previous_updates = prevs  
        
        # self.print_min(self.results[-1])
        
        return self.results
    
    
    def clear_results(self):
        self.results = []
    
    
    def print_min(self, w):
        """Print terminated minimum"""
        
        print('Got to the min:')
        print(f'w0: {w[0]:.2f}\nw1: {w[1]:.2f}\n')