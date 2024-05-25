import numpy as np
import pandas as pd


def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    r = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    r = numerator / denominator if denominator != 0 else 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return r

def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """
    best_features = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    correlations = []
    for column in X.columns:
        column_data = X[column].values
        corr = pearson_correlation(column_data, y)
        correlations.append((column, corr))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    best_features = [column for column, _ in correlations[:n_features]]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return best_features

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = self._add_bias_term(X)
        self.theta = np.random.random(X.shape[1])

        for _ in range(self.n_iter):
            z = np.dot(X, self.theta)
            h = self._sigmoid(z)
            gradient = np.dot(X.T, (h - y))
            self.theta -= self.eta * gradient
            cost = self._compute_cost(h, y)
            self.Js.append(cost)
            self.thetas.append(self.theta.copy())

            # Check for convergence
            if len(self.Js) > 1 and abs(cost - self.Js[-2]) < self.eps:
                break
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = self._add_bias_term(X)
        z = np.dot(X, self.theta)
        h = self._sigmoid(z)
        preds = np.round(h).astype(int)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

    def _add_bias_term(self, X):
        return np.insert(X, 0, 1, axis=1)

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_cost(self, h, y):
        m = len(y)
        return (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))


def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    X, y = _shuffle_data(X, y)
    fold_size = X.shape[0] // folds
    accuracies = []

    for i in range(folds):
        X_train, y_train, X_test, y_test = _split_data(X, y, i, fold_size)

        algo.fit(X_train, y_train)
        y_pred = algo.predict(X_test)

        accuracy = _calculate_accuracy(y_pred, y_test)
        accuracies.append(accuracy)

    cv_accuracy = np.mean(accuracies)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy


def _shuffle_data(X, y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    return X, y


def _split_data(X, y, fold_index, fold_size):
    test_start = fold_index * fold_size
    test_end = (fold_index + 1) * fold_size
    X_test = X[test_start:test_end]
    y_test = y[test_start:test_end]

    train_indices = np.concatenate((np.arange(test_start), np.arange(test_end, X.shape[0])))
    X_train = X[train_indices]
    y_train = y[train_indices]

    return X_train, y_train, X_test, y_test


def _calculate_accuracy(y_pred, y_test):
    return np.mean(y_pred == y_test)


def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    exponent = np.exp(-0.5 * np.square((data - mu) / sigma))
    normalization = 1 / (sigma * np.sqrt(2 * np.pi))
    p = normalization * exponent
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        indexes = np.random.choice(data.shape[0], self.k, replace=False)
        self.weights = np.ones(self.k) / self.k
        self.mus = data[indexes].reshape(self.k)
        self.sigmas = np.random.random(self.k)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        weighted_pdfs = self.weights * norm_pdf(data, self.mus, self.sigmas)
        sum_weighted_pdfs = np.sum(weighted_pdfs, axis=1, keepdims=True)
        self.responsibilities = weighted_pdfs / sum_weighted_pdfs
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.weights = np.mean(self.responsibilities, axis=0)
        responsibilities_sum = np.sum(self.responsibilities, axis=0)
        self.mus = np.sum(self.responsibilities * data.reshape(-1,1), axis=0) / responsibilities_sum
        variance = np.mean(self.responsibilities * np.square(data.reshape(-1, 1) - self.mus), axis=0)
        self.sigmas = np.sqrt(variance / self.weights)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.init_params(data)
        self.costs = self._compute_costs(data)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def _compute_costs(self, data):
        costs = []
        for _ in range(self.n_iter):
            cost = self._compute_cost(data)
            costs.append(cost)
            self.expectation(data)
            self.maximization(data)
            if costs[-1] - cost < self.eps and costs[-1] > cost:
                costs.append(cost)
                break
            costs.append(cost)

        return costs

    def _compute_cost(self, data):
        sum_cost = 0
        cost = self.weights * norm_pdf(data,self.mus,self.sigmas)
        for i in range(len(data)):
            sum_cost = sum_cost + cost[i]
        return -np.sum(np.log(sum_cost))

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    reshaped_data = data.reshape(-1, 1)
    pdf = np.sum(weights * norm_pdf(reshaped_data, mus, sigmas), axis=1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.priors = self._calculate_priors(y)
        self.gaussians = self._init_gaussians_models(X, y)
        # Fit the Gaussian models for each class and each feature
        for class_label, feature_models in self.gaussians.items():
            for feature, model in feature_models.items():
                feature_data = X[y == class_label][:, feature].reshape(-1, 1)
                model.fit(feature_data)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def _init_gaussians_models(self, X, y):
        return {
            class_label: {feature: EM(self.k, random_state=self.random_state)
                          for feature in range(X.shape[1])}
            for class_label in np.unique(y)
        }

    def _calculate_priors(self, y):
        return {
            class_label: np.mean(y == class_label)
            for class_label in np.unique(y)
        }

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        preds = []
        for instance in X:
            posteriors = {
                class_label: self._compute_posterior(instance, class_label)
                for class_label in self.priors.keys()
            }
            predicted_class = max(posteriors, key=posteriors.get)
            preds.append(predicted_class)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

    def _compute_posterior(self, X, class_label):
        likelihood = self._compute_likelihood(X, class_label)
        prior = self.priors[class_label]
        return prior * likelihood

    def _compute_likelihood(self, X, class_label):
        likelihoods = [
            gmm_pdf(X[feature], *self.gaussians[class_label][feature].get_dist_params())
            for feature in range(X.shape[0])
        ]
        return np.prod(likelihoods)


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    logistic_regression = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    logistic_regression.fit(x_train, y_train)
    lor_train_acc = _calculate_accuracy(y_train, logistic_regression.predict(x_train))
    lor_test_acc = _calculate_accuracy(y_test, logistic_regression.predict(x_test))

    naive_bayes = NaiveBayesGaussian(k=k)
    naive_bayes.fit(x_train, y_train)
    bayes_train_acc = _calculate_accuracy(y_train, naive_bayes.predict(x_train))
    bayes_test_acc = _calculate_accuracy(y_test, naive_bayes.predict(x_test))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }