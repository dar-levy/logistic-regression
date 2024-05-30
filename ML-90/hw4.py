from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

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
      Minimal change in the cost to declare convergence.
    random_state : int
      Random number generator seed for random weight initialization.
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
        X = self.bias_trick(X)
        np.random.seed(self.random_state)
        
        # Initialize theta with random values.
        self.theta = np.random.random(X.shape[1])

        # Updating the teta vactor according the gradient descent 
        # Until the difference between the previous cost and the current is less than epsilon,
        # Or we reach n_iter.
        for _ in range(self.n_iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y))
            self.theta = self.theta - self.eta * gradient

            # Calculate cost and check for convergence.
            J = self.cost_function(X, y)
            self.Js.append(J)
            self.thetas.append(self.theta.copy())
            
            if len(self.Js) > 1 and abs(J - self.Js[-2]) < self.eps:
                break

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        X = self.bias_trick(X)
        z = np.dot(X, self.theta)
        h = self.sigmoid(z)
        preds = np.round(h).astype(int)
        return preds

    def bias_trick(self, X):
        """
        Adds a column of 1s as the first feature to account for the bias term.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        return np.insert(X, 0, 1, axis=1)

    def sigmoid(self, z):
        """
        Sigmoid activation function.

        Parameters
        ----------
        z : array-like
        """
        return 1.0 / (1.0 + np.exp(-z))

    def cost_function(self, X, y):
        """
        Compute the cost function.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        z = np.dot(X, self.theta)
        h = self.sigmoid(z)
        epsilon = 1e-5  # small value to avoid log(0).
        J = (-1.0 / len(y)) * (np.dot(y.T, np.log(h + epsilon)) + np.dot((1 - y).T, np.log(1 - h + epsilon)))
        return J



def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation.

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
    np.random.seed(random_state)

    # Shuffle the data.
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    fold_size = X.shape[0] // folds
    accuracies = []

    for i in range(folds):
        # Split data into train and test sets for the current fold.
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]

        train_indices = np.concatenate((np.arange(test_start), np.arange(test_end, X.shape[0])))
        X_train = X[train_indices]
        y_train = y[train_indices]

        # Fit the algorithm on the training data.
        algo.fit(X_train, y_train)

        # Predict labels for the test data.
        y_pred = algo.predict(X_test)

        # Calculate accuracy.
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)

    avg_accuracy = np.mean(accuracies)
    return avg_accuracy


def calculate_accuracy(y_pred, y_test):
    """
    Calculate the accuracy metric.

    Parameters
    ----------
    y_pred : array-like
      Predicted labels.
    y_test : array-like
      True labels.

    Returns
    -------
    accuracy : float
      Accuracy metric.
    """
    accuracy = np.mean(y_pred == y_test)
    return accuracy



def norm_pdf(x, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    
    return(np.exp(np.square((x - mu)) / (-2 * np.square(sigma)))) / (np.sqrt(2 * np.pi * np.square(sigma))) 



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
        self.costs = []

    # Initial guesses for parameters.
    def init_params(self, data):
        """
        Initialize distribution params
        """
        indexes = np.random.choice(data.shape[0], self.k, replace=False)
        self.mus = data[indexes].reshape(self.k)
        self.sigmas = np.random.random_integers(self.k)
        self.weights = np.ones(self.k) / self.k



    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        # Calculate the responsibilities accoording to the formula.
        res = self.weights * norm_pdf(data, self.mus, self.sigmas)
        sum = np.sum(res,axis=1,keepdims=True)
        self.responsibilities=res/sum


    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        # Calculate the distribution params accoording to the formula.
        self.weights = np.mean(self.responsibilities, axis=0)
        self.mus = np.sum(self.responsibilities * data.reshape(-1,1), axis=0) / np.sum(self.responsibilities, axis=0)
        variance = np.mean(self.responsibilities * np.square(data.reshape(-1, 1) - self.mus), axis=0)
        self.sigmas = np.sqrt(variance  / self.weights)


    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
        self.costs.append(self.cost(data))
        
        # Finding the the params for the distribution.
        # We stop when the difference between the previous cost and the current is less than epsilon,
        # Or when we reach n_iter.
        for _ in range(self.n_iter): 
            cost = self.cost(data)
            self.costs.append(cost)
            self.expectation(data)  
            self.maximization(data)  
            if self.costs[-1] - cost < self.eps:
                if self.costs[-1] > cost:
                    self.costs.append(cost)
                    break
            self.costs.append(cost)

    # Calculating the cost of the data.
    def cost(self, data):
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
    pdf = np.sum(weights * norm_pdf(data.reshape(-1,1) , mus , sigmas), axis=1)

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
        self.priors = None
        self.gaussians = None

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
        # Fitting the data.
        self.X = X
        self.y = y
        self.num_Of_Instances = len(X)
        self.priors = {class_Label: len(y[y == class_Label]) / len(y) for class_Label in np.unique(y)}
        self.gaussians = {class_Label: {feature: EM(self.k) for feature in range(X.shape[1])} for class_Label in np.unique(y)}

        for label in self.gaussians.keys():
            for feature in self.gaussians[label].keys():
                self.gaussians[label][feature].fit(X[y == label][:, feature].reshape(-1, 1))

    # Calculate the prior according to the formula.
    def calc_Prior(self, class_label):
        return self.priors[class_label]

    # Calculate the likelihood according to the formula.
    def calc_likelihood(self, X, class_label):
        likelihood = 1
        for feature in range(X.shape[0]):
            weights, mus, sigmas = self.gaussians[class_label][feature].get_dist_params()
            gmm = gmm_pdf(X[feature], weights, mus, sigmas)
            likelihood = likelihood * gmm
        return likelihood

    # Calculate the posterior according to the formula.
    def calc_posterior(self, X, class_label):
        return self.calc_Prior(class_label) * self.calc_likelihood(X, class_label)


    def predict(self, X):
        """
        Return the predicted class labels for a given instance.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]

        Returns
        -------
        preds : array-like, shape = [n_examples]
          Predicted class labels.
        """
        preds = [max([(self.calc_posterior(instance, class_Label), class_Label) for class_Label in self.priors.keys()],
                key=lambda t: t[0])[1] for instance in X]
        return preds



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

    logistic_regression = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    logistic_regression.fit(x_train, y_train)
    lor_train_acc = calculate_accuracy(y_train, logistic_regression.predict(x_train))
    lor_test_acc = calculate_accuracy(y_test, logistic_regression.predict(x_test))

    naive_bayes = NaiveBayesGaussian(k=k)
    naive_bayes.fit(x_train, y_train)
    bayes_train_acc = calculate_accuracy(y_train, naive_bayes.predict(x_train))
    bayes_test_acc = calculate_accuracy(y_test, naive_bayes.predict(x_test))

    return {
        'logistic_regression_train_acc': lor_train_acc,
        'logistic_regression_test_acc': lor_test_acc,
        'naive_bayes_train_acc': bayes_train_acc,
        'naive_bayes_test_acc': bayes_test_acc
    }


def generate_datasets():
    np.random.seed(1991)
    dataset1_features = None
    dataset1_labels = None
    dataset2_features = None
    dataset2_labels = None

    def generate_data(num_instances, means, covariance, labels):
        dataset_features = np.empty((num_instances, 3))
        dataset_labels = np.empty((num_instances))
        gaussian_size = num_instances // len(means)

        for i, mean in enumerate(means):
            label = labels[i]
            points = np.random.multivariate_normal(mean, covariance, gaussian_size)
            dataset_features[i * gaussian_size: (i + 1) * gaussian_size] = points
            dataset_labels[i * gaussian_size: (i + 1) * gaussian_size] = np.full(gaussian_size, label)
        return dataset_features, dataset_labels
    

    dataset1_means = [[0, 0, 0], [4, 4, 4], [12, 12, 12], [18, 18, 18]]
    dataset1_covariance = [[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]]
    dataset1_labels = [1, 0, 0, 1]

    dataset1_features, dataset1_labels = generate_data(5000, dataset1_means, dataset1_covariance, dataset1_labels)
    

    dataset2_means = [[0, 5, 0], [0, 7, 0]]
    dataset2_covariance = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
    dataset2_labels = [0, 1]

    dataset2_features, dataset2_labels = generate_data(5000, dataset2_means, dataset2_covariance, dataset2_labels)
    

    return {
        'dataset1_features': dataset1_features,
        'dataset1_labels': dataset1_labels,
        'dataset2_features': dataset2_features,
        'dataset2_labels': dataset2_labels
    }


