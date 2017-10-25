
import numpy as np
import logging
import json
from utility import * #custom methods for data cleaning

FILE_NAME_TRAIN = 'train.csv' #replace this file name with the train file
FILE_NAME_TEST = 'test.csv' #replace
ALPHA = 0.002
EPOCHS = 50000#keep this greater than or equl to 5000 strictly otherwise you will get an error
MODEL_FILE = 'models/model4'
train_flag = True

logging.basicConfig(filename='output.log',level=logging.DEBUG)

np.set_printoptions(suppress=True)
#################################################################################################
#####################################write the functions here####################################
#################################################################################################
#this function appends 1 to the start of the input X and returns the new array
def appendIntercept(X):
    #steps
    #make a column vector of ones
    #stack this column vector infront of the main X vector using hstack
    #return the new matrix
    col = np.ones((X.shape[0],1))
    X = np.hstack((col,X))
    return X




 #intitial guess of parameters (intialize all to zero)
 #this func takes the number of parameters that is to be fitted and returns a vector of zeros
def initialGuess(n_thetas):
    return np.zeros(n_thetas)



def train(theta, X, y, model):
     J = [] #this array should contain the cost for every iteration so that you can visualize it later when you plot it vs the ith iteration
     #train for the number of epochs you have defined
     m = len(y)
     #your  gradient descent code goes here
     #steps
     #run you gd loop for EPOCHS that you have defined
     for iteration in range(1,EPOCHS):
     	y_predicted = predict(X,theta)
        #calculate the predicted y using your current value of theta
        #cost = costFunc(m,y,y_predicted)
        # calculate cost with that current theta using the costFunc function
        #J.append(cost)
        #append the above cost in J
        gradients = calcGradients(X,y,y_predicted,m)
        #calculate your gradients values using calcGradients function
        theta = makeGradientUpdate(theta,gradients)
        # update the theta using makeGradientUpdate function (don't make a new variable assign it back to theta that you received)
     #print J
     #model['J'] = J
     model['theta'] = list(theta)
     #print theta
     return model


#this function will calculate the total cost and will return it
def costFunc(m,y,y_predicted):
    #takes three parameter as the input m(#training examples), (labeled y), (predicted y)
    #steps
    #apply the formula learnt  
	return np.sum(np.square(y_predicted-y))/(2*m)

def calcGradients(X,y,y_predicted,m):
    #apply the formula , this function will return cost with respect to the gradients
    # basically an numpy array containing n_params
    return np.dot((y_predicted-y),X)

#this function will update the theta and return it
def makeGradientUpdate(theta, grads):
	return theta - ALPHA*grads


#this function will take two paramets as the input
def predict(X,theta):
    exp=1+np.exp(-np.dot(X,theta.T))
    return 1. / exp


########################main function###########################################
def main():
    if(train_flag):
        model = {}
        X_df,y_df = loadData(FILE_NAME_TRAIN)
        X,y, model = normalizeData(X_df, y_df, model)
        X = appendIntercept(X)
        theta = initialGuess(X.shape[1])
        model = train(theta, X, y, model)
        with open(MODEL_FILE,'w') as f:
            f.write(json.dumps(model))
        with open(MODEL_FILE,'r') as f:
            model = json.loads(f.read())
        X_df, y_df = loadData(FILE_NAME_TEST)
        X,y = normalizeTestData(X_df, y_df, model)
        X = appendIntercept(X)    
        accuracy(X,y,model)
        

if __name__ == '__main__':
    main()
