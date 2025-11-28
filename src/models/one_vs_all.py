import numpy as np
from typing import Optional, Dict
from .logistic_regression import LogisticRegression


class OneVsAllClassifier:
    # Multiclass classifier using One-vs-All approach.
    # Trains a separate binary logistic regression classifier for each class.
    
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
        
        self.classifiers = {}  # Dictionary: {class_label: LogisticRegression}
        self.classes = None  # Array of unique class labels
        self.n_classes = 0
        
        # Store loss histories for each classifier
        self.loss_histories = {}
        self.val_loss_histories = {}
    
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None):
        # Train one binary classifier for each class using one-vs-all approach
        # Get unique classes and sort them for consistency
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        if self.verbose:
            print(f"Training One-vs-All classifier for {self.n_classes} classes:")
            print(f"Classes: {self.classes}")
            print("-" * 50)
        
        # Initialize dictionaries
        self.classifiers = {}
        self.loss_histories = {}
        self.val_loss_histories = {}
        
        # Train a binary classifier for each class
        for idx, positive_class in enumerate(self.classes):
            if self.verbose:
                print(f"\n[{idx + 1}/{self.n_classes}] Training classifier for class: {positive_class}")
            
            # Create binary labels: 1 for positive class, 0 for all others
            y_binary = (y == positive_class).astype(int)
            
            # Create validation binary labels if validation set provided
            y_val_binary = None
            if X_val is not None and y_val is not None:
                y_val_binary = (y_val == positive_class).astype(int)
            
            # Create and train binary classifier
            classifier = LogisticRegression(
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                tolerance=self.tolerance,
                regularization=self.regularization,
                lambda_reg=self.lambda_reg,
                verbose=self.verbose
            )
            
            # Train the binary classifier
            classifier.fit(X, y_binary, X_val, y_val_binary)
            
            # Store the classifier
            self.classifiers[positive_class] = classifier
            
            # Store loss histories
            self.loss_histories[positive_class] = classifier.loss_history.copy()
            if X_val is not None and y_val is not None:
                self.val_loss_histories[positive_class] = classifier.val_loss_history.copy()
            
            if self.verbose:
                print(f"  Completed. Final training loss: {classifier.loss_history[-1]:.6f}")
                if X_val is not None and y_val is not None:
                    print(f"  Final validation loss: {classifier.val_loss_history[-1]:.6f}")
        
        if self.verbose:
            print("\n" + "=" * 50)
            print("Training completed for all classes!")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Predict class probabilities for each sample
        if self.classifiers is None or len(self.classifiers) == 0:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, self.n_classes))
        
        # Get probability from each binary classifier
        for idx, class_label in enumerate(self.classes):
            classifier = self.classifiers[class_label]
            # predict_proba returns probability of positive class (this class)
            probabilities[:, idx] = classifier.predict_proba(X)
        
        # Normalize probabilities so they sum to 1 for each sample
        # This is important for one-vs-all approach
        probabilities = probabilities / (probabilities.sum(axis=1, keepdims=True) + 1e-10)
        
        return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predict class labels for samples
        probabilities = self.predict_proba(X)
        # Select class with highest probability
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes[class_indices]
    
    def get_loss_history(self, class_label: Optional[str] = None) -> Dict:
        # Get loss history for a specific class or all classes
        if class_label is not None:
            if class_label not in self.loss_histories:
                raise ValueError(f"Class '{class_label}' not found.")
            return {class_label: self.loss_histories[class_label]}
        return self.loss_histories.copy()
    
    def get_val_loss_history(self, class_label: Optional[str] = None) -> Dict:
        # Get validation loss history for a specific class or all classes
        if class_label is not None:
            if class_label not in self.val_loss_histories:
                raise ValueError(f"Class '{class_label}' not found or no validation data.")
            return {class_label: self.val_loss_histories[class_label]}
        return self.val_loss_histories.copy()
    
    def get_params(self) -> Dict:
        # Get model parameters
        return {
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'tolerance': self.tolerance,
            'regularization': self.regularization,
            'lambda_reg': self.lambda_reg,
            'n_classes': self.n_classes,
            'classes': self.classes.tolist() if self.classes is not None else None
        }
