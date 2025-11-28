import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt


class LogisticRegression:
    # Logistic regression classifier (binary)
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 max_iter: int = 1000,
                 tolerance: float = 1e-6,
                 regularization: str = 'l2',
                 lambda_reg: float = 0.01,
                 verbose: bool = False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.verbose = verbose
        
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.val_loss_history = []
        self.iterations = 0
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        # Sigmoid activation function
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Compute binary cross-entropy loss
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # Add regularization
        if self.regularization == 'l2' and self.weights is not None:
            loss += self.lambda_reg * np.sum(self.weights ** 2)
        elif self.regularization == 'l1' and self.weights is not None:
            loss += self.lambda_reg * np.sum(np.abs(self.weights))
        
        return loss
    
    def _compute_gradients(self, 
                          X: np.ndarray,
                          y_true: np.ndarray,
                          y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
        n_samples = X.shape[0]
        
        # Gradients with respect to loss
        error = y_pred - y_true
        
        # Weight gradients
        weight_grad = np.dot(X.T, error) / n_samples
        
        # Add regularization gradient
        if self.regularization == 'l2':
            weight_grad += 2 * self.lambda_reg * self.weights
        elif self.regularization == 'l1':
            weight_grad += self.lambda_reg * np.sign(self.weights)
        
        # Bias gradient
        bias_grad = np.mean(error)
        
        return weight_grad, bias_grad
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        # Train the model
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Check labels (should be 0 and 1)
        y = np.array(y)
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Labels must be 0 and 1")
        
        prev_loss = float('inf')
        self.loss_history = []
        self.val_loss_history = []
        
        for iteration in range(self.max_iter):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)
            
            # Compute training loss
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Compute validation loss (if validation set provided)
            if X_val is not None and y_val is not None:
                y_val_arr = np.array(y_val)
                z_val = np.dot(X_val, self.weights) + self.bias
                y_pred_val = self._sigmoid(z_val)
                val_loss = self._compute_loss(y_val_arr, y_pred_val)
                self.val_loss_history.append(val_loss)
            
            # Compute gradients
            weight_grad, bias_grad = self._compute_gradients(X, y, y_pred)
            
            # Update parameters
            self.weights -= self.learning_rate * weight_grad
            self.bias -= self.learning_rate * bias_grad
            
            # Convergence check
            if abs(prev_loss - loss) < self.tolerance:
                if self.verbose:
                    print(f"Convergence achieved at iteration {iteration + 1}")
                break
            
            prev_loss = loss
            
            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Loss: {loss:.6f}")
        
        self.iterations = iteration + 1
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Predict probabilities
        if self.weights is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        # Predict class labels
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def plot_loss(self, save_path: Optional[str] = None):
        # Plot loss history
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, label='Training Loss')
        if len(self.val_loss_history) > 0:
            plt.plot(self.val_loss_history, label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
    
    def get_params(self) -> dict:
        # Get model parameters
        return {
            'weights': self.weights,
            'bias': self.bias,
            'learning_rate': self.learning_rate,
            'regularization': self.regularization,
            'lambda_reg': self.lambda_reg,
            'iterations': self.iterations
        }
    
    def set_params(self, weights: np.ndarray, bias: float):
        # Set model parameters
        self.weights = weights
        self.bias = bias