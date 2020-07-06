import numpy as np

class Perceptron (object):
    """"Perceptron Classifier.
    Parameters
    ----------
    
    learn_rate : float
        Learning rate (between 0.0 and 1.0)
    n_itr : int
        Number of iterations through the dataset
    random_seed : int
        Seed for random weigh initialization
    
    Attributes
    -----------
    w_ : ld-array
        Weights 
    errors_ : list
        Number of misclassifications in each epoch
    
    """
    
    def __init__(self, learn_rate = 0.01, n_itr=40, random_seed=1):
        self.learn_rate = learn_rate
        self.n_itr = n_itr
        self.random_seed = random_seed
        
    def fit(self, X, y):
        """ Fit training data.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors
        y: array-like, shape = [n_samples]
            True values
            
        returns
        ----------
        self : object
        
        """
        
        randomgen = np.random.RandomState(self.random.state)
        #normal distribution with standard deviation 0.01
        self.w_ = randomgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        
        for _ in range (self.n_itr):
            errors = 0
            for xi, target in zip(X,y): #aggregate input and output into a tupple
                update = self.learn_rate * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0)
            self.errors_.append(errors)
        return self
        
        def net_value(self, X):
            """Calculate output based on weights and input"""
            return np.dot(X, self.W_[1:]) + self.w_[0] #dot product of input and weights plus bias
        
        def predict(self, X):
            """Return label prediction after each set"""
            return np.where(self.net_value(X) >= 0.0, 1, -1) #if bigger than 0, label = 1 else label = -1