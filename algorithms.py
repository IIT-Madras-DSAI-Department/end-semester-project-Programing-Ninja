# import numpy as np
# import time

# from typing import Optional, Tuple, List


# class MyStandardScaler:
#     """
#     Standardizes features by making them zero mean and unit variance
#     """
#     def __init__(self):
#         self.mean_ = None
#         self.std_ = None
#     def fit(self, X: np.ndarray):
 
#         self.mean_ = np.mean(X, axis=0)
#         # Add a small epsilon to avoid division by zero if a feature has no variance
#         self.std_ = np.std(X, axis=0) + 1e-9 
#     def transform(self, X: np.ndarray) -> np.ndarray:
#         if self.mean_ is None or self.std_ is None:
#             raise ValueError("fit before transforming data")
#         return (X - self.mean_) / self.std_
#     def fit_transform(self, X: np.ndarray) -> np.ndarray:
#         """
#         Does both fit and transform in one step.
#         """
#         self.fit(X)
#         return self.transform(X)
# class PCAModel:
#     def __init__(self, n_components):
#         self.n_components = n_components
#         self.mean = None
#         self.components = None
#         self.explained_variance = None

#     def fit(self, X):
#         X = np.array(X, dtype=float)
#         self.mean = np.mean(X, axis=0)
#         X_centered = X - self.mean
#         cov_matrix = np.cov(X_centered, rowvar=False)
#         eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
#         sorted_idx = np.argsort(eigenvalues)[::-1]
#         self.explained_variance = eigenvalues[sorted_idx][:self.n_components]
#         self.components = eigenvectors[:, sorted_idx][:, :self.n_components]

#     def predict(self, X):

#         if self.mean is None or self.components is None:
#             raise ValueError("The PCA model has not been fitted yet.")
#         X_centered = X - self.mean        
#         return np.dot(X_centered, self.components)

#     def reconstruct(self, X):
#         Z = self.predict(X) 
#         return np.dot(Z, self.components.T) + self.mean

#     def detect_anomalies(self, X, threshold=None, return_errors=False):
#         X_reconstructed = self.reconstruct(X)
#         errors = np.mean((X - X_reconstructed) ** 2, axis=1)
#         if threshold is None:
#             threshold = np.percentile(errors, 95)
            
#         flag = errors > threshold
#         is_anomaly = flag * 1
#         return is_anomaly, errors
    
# class MyKNeighborsClassifier:
#     """
#     KNN classifier using Euclidean distance and majority vote.
#     """
#     def __init__(self, k: int = 5):
#         self.k = k
#         self.X_train_ = None
#         self.y_train_ = None

#     def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple] = None):
#         self.X_train_ = X
#         self.y_train_ = y

#     def predict(self, X_test: np.ndarray) -> np.ndarray:
#         predictions = []
#         for i, x in enumerate(X_test):
#             if (i + 1) % 100 == 0:
#                 print(f"KNN predicting sample {i+1}/{X_test.shape[0]}...")
#             predictions.append(self._predict_one(x))
#         return np.array(predictions)

#     def _predict_one(self, x_test: np.ndarray) -> int:
#         """Helper to predict a single point."""
#         distances = np.sum((self.X_train_ - x_test)**2, axis=1)
#         k_nearest_indices = np.argsort(distances)[:self.k]
#         k_nearest_labels = self.y_train_[k_nearest_indices]        
#         return np.bincount(k_nearest_labels).argmax()

# class MyGaussianNB:
#     """
#     Naive Bayes classifier with Gaussian likelihoods.
#     """
#     def __init__(self):
#         self.classes_ = None
#         self.means_ = {}    
#         self.vars_ = {}    
#         self.priors_ = {}

#     def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple] = None):
#         self.classes_ = np.unique(y)
#         n_samples, n_features = X.shape
        
#         for c in self.classes_:
#             X_c = X[y == c] 
#             self.means_[c] = np.mean(X_c, axis=0)
#             self.vars_[c] = np.var(X_c, axis=0) + 1e-9 # Epsilon for stability
#             self.priors_[c] = X_c.shape[0] / float(n_samples) # P(class)

#     def predict(self, X_test: np.ndarray) -> np.ndarray:
#         posteriors = np.zeros((X_test.shape[0], len(self.classes_)))
        
#         for idx, c in enumerate(self.classes_):
#             prior = np.log(self.priors_[c])
#             mean = self.means_[c]
#             var = self.vars_[c]
#             var_inv = 1 / var
#             log_det = np.sum(np.log(var[var > 0])) # Only sum logs of non-zero variance
            
#             log_pdf_likelihoods = -0.5 * (
#                 np.sum(np.nan_to_num(((X_test - mean)**2) * var_inv, nan=0.0), axis=1) + 
#                 log_det + X_test.shape[1] * np.log(2 * np.pi)
#             )
            
#             posteriors[:, idx] = prior + log_pdf_likelihoods
            
#         return self.classes_[np.argmax(posteriors, axis=1)]

# class MyMultinomialLogisticRegression:
#     """
#     Multinomial Logistic Regression model
#     """
#     def __init__(self, learning_rate: float = 0.01, lambda_p: float = 0.01, 
#                  n_iters: int = 1000, batch_size: int = 128,
#                  patience: int = 10, validation_freq: int = 10):

#         self.lr, self.lambda_p, self.n_iters, self.batch_size = learning_rate, lambda_p, n_iters, batch_size
#         self.patience_ = patience
#         self.validation_freq_ = validation_freq
#         self.W, self.B, self.classes_, self.n_classes_ = None, None, None, 0
#         self.best_W_, self.best_B_, self.best_score_, self.epochs_no_improve_ = None, None, -np.inf, 0

#     def _softmax(self, x: np.ndarray) -> np.ndarray:
#         """Numerically stable softmax."""
#         e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
#         return e_x / np.sum(e_x, axis=1, keepdims=True)

#     def _one_hot(self, y_indices: np.ndarray, n_classes: int) -> np.ndarray:
#         """Creates a one-hot encoding from class indices."""
#         one_hot = np.zeros((y_indices.shape[0], n_classes))
#         one_hot[np.arange(y_indices.shape[0]), y_indices] = 1
#         return one_hot
    
#     def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple] = None):
#         n_samples, n_features = X.shape
#         self.classes_, self.n_classes_ = np.unique(y), len(np.unique(y))
#         self.W, self.B = np.zeros((n_features, self.n_classes_)), np.zeros(self.n_classes_)
#         class_to_index = {c: i for i, c in enumerate(self.classes_)}
#         y_indices = np.array([class_to_index[label] for label in y])

#         print(f"Training Logistic Regression for max {self.n_iters} epochs (Patience={self.patience_})...")
#         for epoch in range(self.n_iters):
#             indices = np.arange(n_samples); np.random.shuffle(indices)
#             X_shuffled, y_shuffled_indices = X[indices], y_indices[indices]
            
#             for i in range(0, n_samples, self.batch_size):
#                 X_batch, y_batch_indices = X_shuffled[i:i+self.batch_size], y_shuffled_indices[i:i+self.batch_size]
#                 batch_n_samples = X_batch.shape[0]                
#                 scores = X_batch @ self.W + self.B
#                 probabilities = self._softmax(scores)
                
#                 y_one_hot = self._one_hot(y_batch_indices, self.n_classes_)
#                 grad_scores = (probabilities - y_one_hot) / batch_n_samples
                
#                 dW = X_batch.T @ grad_scores + (self.lambda_p * self.W)
#                 dB = np.sum(grad_scores, axis=0)
                
#                 self.W -= self.lr * dW
#                 self.B -= self.lr * dB
            
#             # Early Stopping Check
#             if eval_set and (epoch + 1) % self.validation_freq_ == 0:
#                 X_val, y_val = eval_set
#                 y_val_pred = self.predict(X_val)
#                 current_accuracy = np.mean(y_val_pred == y_val)
                
#                 if current_accuracy > self.best_score_:
#                     self.best_score_ = current_accuracy
#                     self.epochs_no_improve_ = 0
#                     self.best_W_ = self.W.copy(); self.best_B_ = self.B.copy()
#                     print(f"Epoch {epoch+1}: New best val_accuracy: {current_accuracy:.4f}")
#                 else:
#                     self.epochs_no_improve_ += 1
#                     print(f"Epoch {epoch+1}: No improvement ({self.epochs_no_improve_}/{self.patience_})")
                
#                 if self.epochs_no_improve_ >= self.patience_:
#                     print(f"Stopping early at epoch {epoch+1}!")
#                     break

#         if self.best_W_ is not None:
#             self.W, self.B = self.best_W_, self.best_B_
            
#     def predict_proba(self, X: np.ndarray) -> np.ndarray:
#         """Predicts class probabilities for X."""
#         scores = X @ self.W + self.B
#         return self._softmax(scores)
        
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         """Predicts the most likely class for X."""
#         return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

# class MyTrueMulticlassSVM:
#     """
#     Multiclass SVM.
#     """
#     def __init__(self, learning_rate: float = 0.01, lambda_p: float = 0.01, 
#                  n_iters: int = 1000, batch_size: int = 128,
#                  patience: int = 10, validation_freq: int = 10):
#         self.lr, self.lambda_p, self.n_iters, self.batch_size = learning_rate, lambda_p, n_iters, batch_size
#         self.patience_ = patience
#         self.validation_freq_ = validation_freq
#         self.W, self.B, self.classes_, self.n_classes_ = None, None, None, 0
#         self.best_W_, self.best_B_, self.best_score_, self.epochs_no_improve_ = None, None, -np.inf, 0

#     def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple] = None):
#         if eval_set is None:
#             raise ValueError("Requires an 'eval_set' for early stopping.")
            
#         n_samples, n_features = X.shape
#         self.classes_, self.n_classes_ = np.unique(y), len(np.unique(y))
#         self.W, self.B = np.zeros((n_features, self.n_classes_)), np.zeros(self.n_classes_)
#         class_to_index = {c: i for i, c in enumerate(self.classes_)}
#         y_indices = np.array([class_to_index[label] for label in y])

#         print(f"Training True SVM for max {self.n_iters} epochs (Patience={self.patience_})...")
#         for epoch in range(self.n_iters):
#             indices = np.arange(n_samples); np.random.shuffle(indices)
#             X_shuffled, y_shuffled = X[indices], y_indices[indices]
            
#             for i in range(0, n_samples, self.batch_size):
#                 X_batch, y_batch = X_shuffled[i:i+self.batch_size], y_shuffled[i:i+self.batch_size]
#                 batch_n_samples = X_batch.shape[0]

#                 scores = X_batch @ self.W + self.B
                
#                 correct_class_scores = scores[np.arange(batch_n_samples), y_batch].reshape(-1, 1)
#                 margins = scores - correct_class_scores + 1.0
#                 margins[np.arange(batch_n_samples), y_batch] = 0
                
#                 violating_class_indices = np.argmax(margins, axis=1)
#                 max_margins = margins[np.arange(batch_n_samples), violating_class_indices]
                
#                 missed_mask = max_margins > 0
#                 num_missed = np.sum(missed_mask)
#                 if num_missed == 0: continue
                
#                 X_missed = X_batch[missed_mask]
#                 y_true_missed = y_batch[missed_mask]
#                 y_violating_missed = violating_class_indices[missed_mask]
                
#                 dW, dB = np.zeros_like(self.W), np.zeros_like(self.B)
#                 np.subtract.at(dW.T, y_true_missed, X_missed)
#                 np.add.at(dW.T, y_violating_missed, X_missed)
#                 np.subtract.at(dB, y_true_missed, 1)
#                 np.add.at(dB, y_violating_missed, 1)
                
#                 dW = (dW / num_missed) + (self.lambda_p * self.W)
#                 dB = dB / num_missed
                
#                 self.W -= self.lr * dW
#                 self.B -= self.lr * dB
            
#             # Early Stopping Check
#             if (epoch + 1) % self.validation_freq_ == 0:
#                 X_val, y_val = eval_set
#                 y_val_pred = self.predict(X_val)
#                 current_accuracy = np.mean(y_val_pred == y_val)
                
#                 if current_accuracy > self.best_score_:
#                     self.best_score_ = current_accuracy
#                     self.epochs_no_improve_ = 0
#                     self.best_W_ = self.W.copy(); self.best_B_ = self.B.copy()
#                     print(f"Epoch {epoch+1}: New best val_accuracy: {current_accuracy:.4f}")
#                 else:
#                     self.epochs_no_improve_ += 1
#                     print(f"Epoch {epoch+1}: No improvement ({self.epochs_no_improve_}/{self.patience_})")
                
#                 if self.epochs_no_improve_ >= self.patience_:
#                     print(f"Stopping early at epoch {epoch+1}!")
#                     break

#         if self.best_W_ is not None:
#             self.W, self.B = self.best_W_, self.best_B_

#     def predict(self, X: np.ndarray) -> np.ndarray:
#         """Predicts the class with the highest score."""
#         scores = X @ self.W + self.B
#         return self.classes_[np.argmax(scores, axis=1)]

# class _DecisionTreeLeaf:
#     def __init__(self, y: np.ndarray):
#         self.predictions = np.bincount(y)

# class _DecisionTreeNode:
#     """A Node that splits the data."""
#     def __init__(self, feature_index: int, threshold: float, 
#                  left_child, right_child):
#         self.feature_index = feature_index
#         self.threshold = threshold
#         self.left_child = left_child
#         self.right_child = right_child

# class MyDecisionTreeClassifier:
#     """
#     Classification tree using Gini Impurity.
#     """
#     def __init__(self, max_depth: int = 5, min_samples_split: int = 2, 
#                  max_features: Optional[str] = None):
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.max_features = max_features
#         self.root_ = None
#         self.n_features_ = 0
    
#     def fit(self, X: np.ndarray, y: np.ndarray):
#         self.n_features_ = X.shape[1]
        
#         if self.max_features is None:
#             self.n_features_to_sample_ = self.n_features_
#         elif self.max_features == 'sqrt':
#             self.n_features_to_sample_ = int(np.sqrt(self.n_features_))
#         else:
#             self.n_features_to_sample_ = int(self.max_features)
            
#         self.root_ = self._grow_tree(X, y, depth=0)
    
#     def _gini_impurity(self, y: np.ndarray) -> float:
#         if y.shape[0] == 0:
#             return 0
#         counts = np.bincount(y)
#         probabilities = counts / y.shape[0]
#         return 1.0 - np.sum(probabilities**2)
        
#     def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple:
#         n_samples, n_features = X.shape
#         if n_samples < self.min_samples_split:
#             return None, None
            
#         parent_gini = self._gini_impurity(y)
#         best_gain = -1.0
#         best_feature_index = None
#         best_threshold = None
        
#         feature_indices = np.random.choice(n_features, self.n_features_to_sample_, replace=False)
        
#         for feature_index in feature_indices:
#             thresholds = np.unique(np.quantile(X[:, feature_index], q=np.linspace(0.01, 0.99, 10)))
            
#             for threshold in thresholds:
#                 left_indices = X[:, feature_index] <= threshold
#                 right_indices = ~left_indices
#                 y_left, y_right = y[left_indices], y[right_indices]
                
#                 if y_left.shape[0] == 0 or y_right.shape[0] == 0:
#                     continue
                    
#                 p_left = y_left.shape[0] / n_samples
#                 p_right = y_right.shape[0] / n_samples
#                 weighted_gini = (p_left * self._gini_impurity(y_left)) + (p_right * self._gini_impurity(y_right))
                
#                 gini_gain = parent_gini - weighted_gini
                
#                 if gini_gain > best_gain:
#                     best_gain, best_feature_index, best_threshold = gini_gain, feature_index, threshold
                    
#         return best_feature_index, best_threshold

#     def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
#         """Recursively builds the tree."""
#         if (depth >= self.max_depth or
#             y.shape[0] < self.min_samples_split or
#             len(np.unique(y)) == 1):
#             return _DecisionTreeLeaf(y)
            
#         best_feature_index, best_threshold = self._find_best_split(X, y)
        
#         if best_feature_index is None:
#             return _DecisionTreeLeaf(y)
            
#         left_indices = X[:, best_feature_index] <= best_threshold
#         right_indices = ~left_indices
        
#         if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
#             return _DecisionTreeLeaf(y)

#         left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
#         right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
#         return _DecisionTreeNode(best_feature_index, best_threshold, left_child, right_child)

#     def predict(self, X_test: np.ndarray) -> np.ndarray:
#         return np.array([self._predict_one(x) for x in X_test])
        
#     def _predict_one(self, x: np.ndarray) -> int:
#         node = self.root_
#         while isinstance(node, _DecisionTreeNode):
#             if x[node.feature_index] <= node.threshold:
#                 node = node.left_child
#             else:
#                 node = node.right_child
        
#         return np.argmax(node.predictions)

# class MyRandomForestClassifier:
#     def __init__(self, n_estimators: int = 100, max_depth: int = 10, 
#                  min_samples_split: int = 2, max_features: str = 'sqrt'):
#         self.n_estimators = n_estimators
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.max_features = max_features
#         self.trees_ = []
    
#     def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         n_samples = X.shape[0]
#         indices = np.random.choice(n_samples, n_samples, replace=True)
#         return X[indices], y[indices]
    
#     def _majority_vote(self, all_tree_preds_for_one_sample: np.ndarray) -> int:
#         return np.bincount(all_tree_preds_for_one_sample).argmax()

#     def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple] = None):
#         self.trees_ = []
#         for i in range(self.n_estimators):
#             if (i + 1) % 10 == 0:
#                 print(f"Training tree {i+1}/{self.n_estimators}")
            
#             tree = MyDecisionTreeClassifier(
#                 max_depth=self.max_depth,
#                 min_samples_split=self.min_samples_split,
#                 max_features=self.max_features
#             )
            
#             X_sample, y_sample = self._bootstrap_sample(X, y)
#             tree.fit(X_sample, y_sample)
#             self.trees_.append(tree)

#     def predict(self, X_test: np.ndarray) -> np.ndarray:
#         all_tree_preds = np.array([tree.predict(X_test) for tree in self.trees_])
#         all_tree_preds = all_tree_preds.T 
#         predictions = [self._majority_vote(sample_preds) for sample_preds in all_tree_preds]
#         return np.array(predictions)

# class MyVotingClassifier:
#     """
#     Takes the predictions from multiple
#     models and predicts the class that received the most votes.
#     """
#     def __init__(self):
#         pass

#     def predict(self, list_of_prediction_arrays: List[np.ndarray]) -> np.ndarray:
#         stacked_preds = np.column_stack(list_of_prediction_arrays)
        
#         predictions = np.apply_along_axis(
#             lambda x: np.bincount(x).argmax(), 
#             axis=1, 
#             arr=stacked_preds.astype(int) 
#         )
#         return predictions

# class _BinarySVM:

#     def __init__(self, learning_rate=0.01, lambda_p=0.0001, n_iters=1000, batch_size=64):
#         self.lr, self.lambda_p, self.n_iters, self.batch_size = learning_rate, lambda_p, n_iters, batch_size
#         self.w, self.b = None, 0
#     def fit(self, X, y):
#         y_ = np.where(y > 0, 1, -1)
#         n_samples, n_features = X.shape
#         self.w = np.zeros(n_features)
#         for _ in range(self.n_iters):
#             indices = np.arange(n_samples); np.random.shuffle(indices)
#             X_shuffled, y_shuffled = X[indices], y_[indices]
#             for i in range(0, n_samples, self.batch_size):
#                 X_batch, y_batch = X_shuffled[i:i+self.batch_size], y_shuffled[i:i+self.batch_size]
#                 conditions = y_batch * (np.dot(X_batch, self.w) - self.b)
#                 missed_mask = conditions < 1
#                 dw_reg = self.lambda_p * self.w
#                 dw_data = -np.dot(y_batch[missed_mask].T, X_batch[missed_mask])
#                 num_missed = np.sum(missed_mask)
#                 if num_missed > 0:
#                     dw = (dw_data / num_missed) + dw_reg
#                     db_data = -np.sum(y_batch[missed_mask])
#                     db = db_data / num_missed
#                 else:
#                     dw, db = dw_reg, 0
#                 self.w -= self.lr * dw; self.b -= self.lr * db
#     def predict_score(self, X):
#         return np.dot(X, self.w) - self.b
#     def predict(self, X):
#         return np.sign(self.predict_score(X))

# class MyMulticlassSVM_OvR:
#     def __init__(self, **kwargs):
#         self.kwargs = kwargs
#         self.classifiers = []
#         self.classes_ = None
#     def fit(self, X, y):
#         self.classes_ = np.unique(y)
#         for c in self.classes_:
#             y_binary = np.where(y == c, 1, -1)
#             svm = _BinarySVM(**self.kwargs)
#             svm.fit(X, y_binary)
#             self.classifiers.append(svm)
#     def predict(self, X):
#         scores = np.column_stack([svm.predict_score(X) for svm in self.classifiers])
#         return self.classes_[np.argmax(scores, axis=1)]

# class MyMulticlassSVM_OvO:
#     def __init__(self, **kwargs):
#         self.kwargs = kwargs
#         self.classifiers = {}
#         self.classes_ = None
#     def fit(self, X, y):
#         self.classes_ = np.unique(y)
#         class_pairs = list(itertools.combinations(self.classes_, 2))
#         for (c1, c2) in class_pairs:
#             mask = (y == c1) | (y == c2)
#             X_pair, y_pair = X[mask], y[mask]
#             y_binary = np.where(y_pair == c1, 1, -1)
#             svm = _BinarySVM(**self.kwargs)
#             svm.fit(X_pair, y_binary)
#             self.classifiers[(c1, c2)] = svm
#     def predict(self, X):
#         n_samples = X.shape[0]
#         vote_counts = np.zeros((n_samples, len(self.classes_)))
#         class_to_index = {c: i for i, c in enumerate(self.classes_)}
#         for (c1, c2), svm in self.classifiers.items():
#             preds = svm.predict(X) # +1 for c1, -1 for c2
#             c1_idx, c2_idx = class_to_index[c1], class_to_index[c2]
#             for i in range(n_samples):
#                 if preds[i] == 1:
#                     vote_counts[i, c1_idx] += 1
#                 else:
#                     vote_counts[i, c2_idx] += 1
#         return self.classes_[np.argmax(vote_counts, axis=1)]

import numpy as np
import time
import itertools  # <--- ADDED THIS IMPORT

from typing import Optional, Tuple, List


class MyStandardScaler:
    """
    Standardizes features by making them zero mean and unit variance
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    def fit(self, X: np.ndarray):
 
        self.mean_ = np.mean(X, axis=0)
        # Add a small epsilon to avoid division by zero if a feature has no variance
        self.std_ = np.std(X, axis=0) + 1e-9 
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("fit before transforming data")
        return (X - self.mean_) / self.std_
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Does both fit and transform in one step.
        """
        self.fit(X)
        return self.transform(X)
class PCAModel:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        X = np.array(X, dtype=float)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        self.explained_variance = eigenvalues[sorted_idx][:self.n_components]
        self.components = eigenvectors[:, sorted_idx][:, :self.n_components]

    def predict(self, X):

        if self.mean is None or self.components is None:
            raise ValueError("The PCA model has not been fitted yet.")
        X_centered = X - self.mean        
        return np.dot(X_centered, self.components)

    def reconstruct(self, X):
        Z = self.predict(X) 
        return np.dot(Z, self.components.T) + self.mean

    def detect_anomalies(self, X, threshold=None, return_errors=False):
        X_reconstructed = self.reconstruct(X)
        errors = np.mean((X - X_reconstructed) ** 2, axis=1)
        if threshold is None:
            threshold = np.percentile(errors, 95)
            
        flag = errors > threshold
        is_anomaly = flag * 1
        return is_anomaly, errors
    
class MyKNeighborsClassifier:
    """
    KNN classifier using Euclidean distance and majority vote.
    """
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple] = None):
        self.X_train_ = X
        self.y_train_ = y

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predictions = []
        for i, x in enumerate(X_test):
            if (i + 1) % 100 == 0:
                print(f"KNN predicting sample {i+1}/{X_test.shape[0]}...")
            predictions.append(self._predict_one(x))
        return np.array(predictions)

    def _predict_one(self, x_test: np.ndarray) -> int:
        """Helper to predict a single point."""
        distances = np.sum((self.X_train_ - x_test)**2, axis=1)
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train_[k_nearest_indices]        
        return np.bincount(k_nearest_labels).argmax()

class MyGaussianNB:
    """
    Naive Bayes classifier with Gaussian likelihoods.
    """
    def __init__(self):
        self.classes_ = None
        self.means_ = {}    
        self.vars_ = {}    
        self.priors_ = {}

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple] = None):
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        
        for c in self.classes_:
            X_c = X[y == c] 
            self.means_[c] = np.mean(X_c, axis=0)
            self.vars_[c] = np.var(X_c, axis=0) + 1e-9 # Epsilon for stability
            self.priors_[c] = X_c.shape[0] / float(n_samples) # P(class)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        posteriors = np.zeros((X_test.shape[0], len(self.classes_)))
        
        for idx, c in enumerate(self.classes_):
            prior = np.log(self.priors_[c])
            mean = self.means_[c]
            var = self.vars_[c]
            var_inv = 1 / var
            log_det = np.sum(np.log(var[var > 0])) # Only sum logs of non-zero variance
            
            log_pdf_likelihoods = -0.5 * (
                np.sum(np.nan_to_num(((X_test - mean)**2) * var_inv, nan=0.0), axis=1) + 
                log_det + X_test.shape[1] * np.log(2 * np.pi)
            )
            
            posteriors[:, idx] = prior + log_pdf_likelihoods
            
        return self.classes_[np.argmax(posteriors, axis=1)]

class MyMultinomialLogisticRegression:
    """
    Multinomial Logistic Regression model
    """
    def __init__(self, learning_rate: float = 0.01, lambda_p: float = 0.01, 
                 n_iters: int = 1000, batch_size: int = 128,
                 patience: int = 10, validation_freq: int = 10):

        self.lr, self.lambda_p, self.n_iters, self.batch_size = learning_rate, lambda_p, n_iters, batch_size
        self.patience_ = patience
        self.validation_freq_ = validation_freq
        self.W, self.B, self.classes_, self.n_classes_ = None, None, None, 0
        self.best_W_, self.best_B_, self.best_score_, self.epochs_no_improve_ = None, None, -np.inf, 0

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def _one_hot(self, y_indices: np.ndarray, n_classes: int) -> np.ndarray:
        """Creates a one-hot encoding from class indices."""
        one_hot = np.zeros((y_indices.shape[0], n_classes))
        one_hot[np.arange(y_indices.shape[0]), y_indices] = 1
        return one_hot
    
    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple] = None):
        n_samples, n_features = X.shape
        self.classes_, self.n_classes_ = np.unique(y), len(np.unique(y))
        self.W, self.B = np.zeros((n_features, self.n_classes_)), np.zeros(self.n_classes_)
        class_to_index = {c: i for i, c in enumerate(self.classes_)}
        y_indices = np.array([class_to_index[label] for label in y])

        print(f"Training Logistic Regression for max {self.n_iters} epochs (Patience={self.patience_})...")
        for epoch in range(self.n_iters):
            indices = np.arange(n_samples); np.random.shuffle(indices)
            X_shuffled, y_shuffled_indices = X[indices], y_indices[indices]
            
            for i in range(0, n_samples, self.batch_size):
                X_batch, y_batch_indices = X_shuffled[i:i+self.batch_size], y_shuffled_indices[i:i+self.batch_size]
                batch_n_samples = X_batch.shape[0]                
                scores = X_batch @ self.W + self.B
                probabilities = self._softmax(scores)
                
                y_one_hot = self._one_hot(y_batch_indices, self.n_classes_)
                grad_scores = (probabilities - y_one_hot) / batch_n_samples
                
                dW = X_batch.T @ grad_scores + (self.lambda_p * self.W)
                dB = np.sum(grad_scores, axis=0)
                
                self.W -= self.lr * dW
                self.B -= self.lr * dB
            
            # Early Stopping Check
            if eval_set and (epoch + 1) % self.validation_freq_ == 0:
                X_val, y_val = eval_set
                y_val_pred = self.predict(X_val)
                current_accuracy = np.mean(y_val_pred == y_val)
                
                if current_accuracy > self.best_score_:
                    self.best_score_ = current_accuracy
                    self.epochs_no_improve_ = 0
                    self.best_W_ = self.W.copy(); self.best_B_ = self.B.copy()
                    print(f"Epoch {epoch+1}: New best val_accuracy: {current_accuracy:.4f}")
                else:
                    self.epochs_no_improve_ += 1
                    print(f"Epoch {epoch+1}: No improvement ({self.epochs_no_improve_}/{self.patience_})")
                
                if self.epochs_no_improve_ >= self.patience_:
                    print(f"Stopping early at epoch {epoch+1}!")
                    break

        if self.best_W_ is not None:
            self.W, self.B = self.best_W_, self.best_B_
            
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for X."""
        scores = X @ self.W + self.B
        return self._softmax(scores)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the most likely class for X."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

class MyTrueMulticlassSVM:
    """
    Multiclass SVM.
    """
    def __init__(self, learning_rate: float = 0.01, lambda_p: float = 0.01, 
                 n_iters: int = 1000, batch_size: int = 128,
                 patience: int = 10, validation_freq: int = 10):
        self.lr, self.lambda_p, self.n_iters, self.batch_size = learning_rate, lambda_p, n_iters, batch_size
        self.patience_ = patience
        self.validation_freq_ = validation_freq
        self.W, self.B, self.classes_, self.n_classes_ = None, None, None, 0
        self.best_W_, self.best_B_, self.best_score_, self.epochs_no_improve_ = None, None, -np.inf, 0

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple] = None):
        if eval_set is None:
            raise ValueError("Requires an 'eval_set' for early stopping.")
            
        n_samples, n_features = X.shape
        self.classes_, self.n_classes_ = np.unique(y), len(np.unique(y))
        self.W, self.B = np.zeros((n_features, self.n_classes_)), np.zeros(self.n_classes_)
        class_to_index = {c: i for i, c in enumerate(self.classes_)}
        y_indices = np.array([class_to_index[label] for label in y])

        print(f"Training True SVM for max {self.n_iters} epochs (Patience={self.patience_})...")
        for epoch in range(self.n_iters):
            indices = np.arange(n_samples); np.random.shuffle(indices)
            X_shuffled, y_shuffled = X[indices], y_indices[indices]
            
            for i in range(0, n_samples, self.batch_size):
                X_batch, y_batch = X_shuffled[i:i+self.batch_size], y_shuffled[i:i+self.batch_size]
                batch_n_samples = X_batch.shape[0]

                scores = X_batch @ self.W + self.B
                
                correct_class_scores = scores[np.arange(batch_n_samples), y_batch].reshape(-1, 1)
                margins = scores - correct_class_scores + 1.0
                margins[np.arange(batch_n_samples), y_batch] = 0
                
                violating_class_indices = np.argmax(margins, axis=1)
                max_margins = margins[np.arange(batch_n_samples), violating_class_indices]
                
                missed_mask = max_margins > 0
                num_missed = np.sum(missed_mask)
                if num_missed == 0: continue
                
                X_missed = X_batch[missed_mask]
                y_true_missed = y_batch[missed_mask]
                y_violating_missed = violating_class_indices[missed_mask]
                
                dW, dB = np.zeros_like(self.W), np.zeros_like(self.B)
                np.subtract.at(dW.T, y_true_missed, X_missed)
                np.add.at(dW.T, y_violating_missed, X_missed)
                np.subtract.at(dB, y_true_missed, 1)
                np.add.at(dB, y_violating_missed, 1)
                
                dW = (dW / num_missed) + (self.lambda_p * self.W)
                dB = dB / num_missed
                
                self.W -= self.lr * dW
                self.B -= self.lr * dB
            
            # Early Stopping Check
            if (epoch + 1) % self.validation_freq_ == 0:
                X_val, y_val = eval_set
                y_val_pred = self.predict(X_val)
                current_accuracy = np.mean(y_val_pred == y_val)
                
                if current_accuracy > self.best_score_:
                    self.best_score_ = current_accuracy
                    self.epochs_no_improve_ = 0
                    self.best_W_ = self.W.copy(); self.best_B_ = self.B.copy()
                    print(f"Epoch {epoch+1}: New best val_accuracy: {current_accuracy:.4f}")
                else:
                    self.epochs_no_improve_ += 1
                    print(f"Epoch {epoch+1}: No improvement ({self.epochs_no_improve_}/{self.patience_})")
                
                if self.epochs_no_improve_ >= self.patience_:
                    print(f"Stopping early at epoch {epoch+1}!")
                    break

        if self.best_W_ is not None:
            self.W, self.B = self.best_W_, self.best_B_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the class with the highest score."""
        scores = X @ self.W + self.B
        return self.classes_[np.argmax(scores, axis=1)]

class _DecisionTreeLeaf:
    def __init__(self, y: np.ndarray):
        self.predictions = np.bincount(y)

class _DecisionTreeNode:
    """A Node that splits the data."""
    def __init__(self, feature_index: int, threshold: float, 
                 left_child, right_child):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child

class MyDecisionTreeClassifier:
    """
    Classification tree using Gini Impurity.
    """
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, 
                 max_features: Optional[str] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root_ = None
        self.n_features_ = 0
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_features_ = X.shape[1]
        
        if self.max_features is None:
            self.n_features_to_sample_ = self.n_features_
        elif self.max_features == 'sqrt':
            self.n_features_to_sample_ = int(np.sqrt(self.n_features_))
        else:
            self.n_features_to_sample_ = int(self.max_features)
            
        self.root_ = self._grow_tree(X, y, depth=0)
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        if y.shape[0] == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / y.shape[0]
        return 1.0 - np.sum(probabilities**2)
        
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None
            
        parent_gini = self._gini_impurity(y)
        best_gain = -1.0
        best_feature_index = None
        best_threshold = None
        
        feature_indices = np.random.choice(n_features, self.n_features_to_sample_, replace=False)
        
        for feature_index in feature_indices:
            thresholds = np.unique(np.quantile(X[:, feature_index], q=np.linspace(0.01, 0.99, 10)))
            
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = ~left_indices
                y_left, y_right = y[left_indices], y[right_indices]
                
                if y_left.shape[0] == 0 or y_right.shape[0] == 0:
                    continue
                    
                p_left = y_left.shape[0] / n_samples
                p_right = y_right.shape[0] / n_samples
                weighted_gini = (p_left * self._gini_impurity(y_left)) + (p_right * self._gini_impurity(y_right))
                
                gini_gain = parent_gini - weighted_gini
                
                if gini_gain > best_gain:
                    best_gain, best_feature_index, best_threshold = gini_gain, feature_index, threshold
                    
        return best_feature_index, best_threshold

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        """Recursively builds the tree."""
        if (depth >= self.max_depth or
            y.shape[0] < self.min_samples_split or
            len(np.unique(y)) == 1):
            return _DecisionTreeLeaf(y)
            
        best_feature_index, best_threshold = self._find_best_split(X, y)
        
        if best_feature_index is None:
            return _DecisionTreeLeaf(y)
            
        left_indices = X[:, best_feature_index] <= best_threshold
        right_indices = ~left_indices
        
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return _DecisionTreeLeaf(y)

        left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        return _DecisionTreeNode(best_feature_index, best_threshold, left_child, right_child)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x) for x in X_test])
        
    def _predict_one(self, x: np.ndarray) -> int:
        node = self.root_
        while isinstance(node, _DecisionTreeNode):
            if x[node.feature_index] <= node.threshold:
                node = node.left_child
            else:
                node = node.right_child
        
        return np.argmax(node.predictions)

class MyRandomForestClassifier:
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, 
                 min_samples_split: int = 2, max_features: str = 'sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees_ = []
    
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def _majority_vote(self, all_tree_preds_for_one_sample: np.ndarray) -> int:
        return np.bincount(all_tree_preds_for_one_sample).argmax()

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple] = None):
        self.trees_ = []
        for i in range(self.n_estimators):
            if (i + 1) % 10 == 0:
                print(f"Training tree {i+1}/{self.n_estimators}")
            
            tree = MyDecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        all_tree_preds = np.array([tree.predict(X_test) for tree in self.trees_])
        all_tree_preds = all_tree_preds.T 
        predictions = [self._majority_vote(sample_preds) for sample_preds in all_tree_preds]
        return np.array(predictions)

class MyVotingClassifier:
    """
    Takes the predictions from multiple
    models and predicts the class that received the most votes.
    """
    def __init__(self):
        pass

    def predict(self, list_of_prediction_arrays: List[np.ndarray]) -> np.ndarray:
        stacked_preds = np.column_stack(list_of_prediction_arrays)
        
        predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 
            axis=1, 
            arr=stacked_preds.astype(int) 
        )
        return predictions

class _BinarySVM:

    def __init__(self, learning_rate=0.01, lambda_p=0.0001, n_iters=1000, batch_size=64):
        self.lr, self.lambda_p, self.n_iters, self.batch_size = learning_rate, lambda_p, n_iters, batch_size
        self.w, self.b = None, 0
    def fit(self, X, y):
        y_ = np.where(y > 0, 1, -1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        for _ in range(self.n_iters):
            indices = np.arange(n_samples); np.random.shuffle(indices)
            X_shuffled, y_shuffled = X[indices], y_[indices]
            for i in range(0, n_samples, self.batch_size):
                X_batch, y_batch = X_shuffled[i:i+self.batch_size], y_shuffled[i:i+self.batch_size]
                conditions = y_batch * (np.dot(X_batch, self.w) - self.b)
                missed_mask = conditions < 1
                dw_reg = self.lambda_p * self.w
                dw_data = -np.dot(y_batch[missed_mask].T, X_batch[missed_mask])
                num_missed = np.sum(missed_mask)
                if num_missed > 0:
                    dw = (dw_data / num_missed) + dw_reg
                    db_data = -np.sum(y_batch[missed_mask])
                    db = db_data / num_missed
                else:
                    dw, db = dw_reg, 0
                self.w -= self.lr * dw; self.b -= self.lr * db
    def predict_score(self, X):
        return np.dot(X, self.w) - self.b
    def predict(self, X):
        return np.sign(self.predict_score(X))

class MyMulticlassSVM_OvR:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.classifiers = []
        self.classes_ = None
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            y_binary = np.where(y == c, 1, -1)
            svm = _BinarySVM(**self.kwargs)
            svm.fit(X, y_binary)
            self.classifiers.append(svm)
    def predict(self, X):
        scores = np.column_stack([svm.predict_score(X) for svm in self.classifiers])
        return self.classes_[np.argmax(scores, axis=1)]

class MyMulticlassSVM_OvO:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.classifiers = {}
        self.classes_ = None
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        class_pairs = list(itertools.combinations(self.classes_, 2))
        for (c1, c2) in class_pairs:
            mask = (y == c1) | (y == c2)
            X_pair, y_pair = X[mask], y[mask]
            y_binary = np.where(y_pair == c1, 1, -1)
            svm = _BinarySVM(**self.kwargs)
            svm.fit(X_pair, y_binary)
            self.classifiers[(c1, c2)] = svm
    def predict(self, X):
        n_samples = X.shape[0]
        vote_counts = np.zeros((n_samples, len(self.classes_)))
        class_to_index = {c: i for i, c in enumerate(self.classes_)}
        for (c1, c2), svm in self.classifiers.items():
            preds = svm.predict(X) # +1 for c1, -1 for c2
            c1_idx, c2_idx = class_to_index[c1], class_to_index[c2]
            for i in range(n_samples):
                if preds[i] == 1:
                    vote_counts[i, c1_idx] += 1
                else:
                    vote_counts[i, c2_idx] += 1
        return self.classes_[np.argmax(vote_counts, axis=1)]


class MyEnsembleClassifier:
    def __init__(self, estimators: List):
        self.estimators_ = estimators
        self.voter_ = MyVotingClassifier() # Uses the existing voting class

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple] = None):
        print(f"Fitting {len(self.estimators_)} base models for the ensemble...")
        for model in self.estimators_:
            model_name = model.__class__.__name__
            print(f"Fitting base model: {model_name} ---")
            start_time = time.time()
            try:
                model.fit(X, y, eval_set=eval_set)
            except TypeError as e:
                if 'unexpected keyword argument' in str(e) or 'eval_set' in str(e):
                    model.fit(X, y)
                else:
                    raise e
            fit_time = time.time() - start_time
            print(f"Completed {model_name} in {fit_time:.2f}s")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        all_predictions = []
        for model in self.estimators_:
            model_name = model.__class__.__name__
            start_time = time.time()
            preds = model.predict(X_test)
            all_predictions.append(preds)
            pred_time = time.time() - start_time

        print("Combining predictions with MyVotingClassifier")
        return self.voter_.predict(all_predictions)