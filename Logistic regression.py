"""
#note that number of rows of X = no of features
#note that number of cols of X = no of examples

#note that number of rows of y = no of outputs/classes
#note that number of cols of y = no of examples
"""

import numpy as np

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))    
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    """
    #In case error = (prediction-y_true) where prediction is probability calculated using
    #(- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A))) so
    #dw = derror/prediction*dprediction/A*dA/w
    #db = derror/prediction*dprediction/A*dA/b
    #dw = (-1/m*((Y*(1/A))+((1-Y)/(1-A))*(-1)))*((A)*(1-A))*(-X)  
    #sum over all samples
    #then take norm of it
    #or simply
    dw = (1 / m) * np.dot(X, (A - Y).T)
    #db = (-1/m*((Y*(1/A))+((1-Y)/(1-A))*(-1)))*((A)*(1-A))*(1) 
    #sum over all samples
    #then take norm of it
    #or simply
    db = (1 / m) * np.sum(A - Y)

    in case of gradient descent
    w = w+(eta*dw)
    
    in case of gradient descent with momentum
    alpha = 0.9
    nu = (alpha*nu)+(eta*dw)
    w = w+nu
    
    in case of RMSProp
    G = (alpha*G)+((1-alpha)*(dw**2)) 
    w = w+((eta/np.sqrt(G+eps))*dw)
    """
    #In case error = (prediction-y_true) where prediction is probability calculated using
    #(- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A))) so
    #dw = derror/prediction*dprediction/A*dA/w
    #db = derror/prediction*dprediction/A*dA/b
    #dw = (-1/m*((Y*(1/A))+((1-Y)/(1-A))*(-1)))*((A)*(1-A))*(-X)  
    #sum over all samples
    #then take norm of it
    #or simply
    dw = (1 / m) * np.dot(X, (A - Y).T)
    #db = (-1/m*((Y*(1/A))+((1-Y)/(1-A))*(-1)))*((A)*(1-A))*(1) 
    #sum over all samples
    #then take norm of it
    #or simply
    db = (1 / m) * np.sum(A - Y)


    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):    
    costs = []
    
    for i in range(num_iterations):
                
        # Cost and gradient calculation (≈ 1-4 lines of code)
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        w = w - learning_rate * dw  # need to broadcast
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        # Convert probabilities a[0,i] to actual predictions p[0,i]
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    
    assert(Y_prediction.shape == (1, m))    
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):  
    
    """
    X_train -- training set represented by a numpy array of shape (features, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (no of outputs, m_train)
    X_test -- test set represented by a numpy array of shape (features, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (no of outputs, m_test)
    """
    
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0]) #note that number of rows of X = no of features

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    
    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


######################################################################################
