#using method of steepest descent on a single perceptron network
#to map I/O pairs x1 = (x11, x12), x2 = (x21, x22), ..., to desired output values, d1, d2, ..., at node 3, respectively.

import numpy as np
np.set_printoptions(precision=4, suppress=True)
import time


# implements a linear activity function of a single node with two inputs
def activity_function(x1, x2, w1j, w2j, bias_j):
    A_j = x1*w1j + x2*w2j + bias_j
    return A_j


#implements the softmax activation function
def softmax_activation_function(A):
    return np.log(1 + np.exp(A))


#implements the sigmoid activation function
def sigmoid_activation_function(A):
    return 1/(1.0 + np.exp(-1*A))


#implements the sigmoid activation function
def activation(A):
    return 1/(1.0 + np.exp(-1*A))



#training set data format [LAC, SOW, TACA] #train on LAC & SOW to predict TACA score of 1
xTrain = [  [0.90, 0.87, 1],
            [1.31, 0.75, 1],
            [2.48, 1.14, 0],
            [0.41, 1.87, 0],
            [2.45, 0.52, 0],
            [2.54, 2.97, 1],
            [0.07, 0.09, 1],
            [1.32, 1.96, 0],
            [0.94, 0.34, 1],
            [1.75, 2.21, 0]]

#testing set data format [LAC, SOW, TACA]
xTest = [   [1.81, 1.02, 0],
            [2.36, 1.60, 0],
            [2.17, 2.08, 1],
            [2.85, 2.91, 1],
            [1.05, 1.93, 0],
            [2.32, 1.73, 0],
            [1.86, 1.31, 0],
            [1.45, 2.19, 0],
            [0.28, 0.71, 1],
            [2.49, 1.52, 0]]



#initial inputs, weights, bias, eta. Desired value is d.
w1j = np.random.rand()
w2j = np.random.rand()
bias_j = np.random.rand()
eta = 1.0
d_j = 0.15
numEpochs= 30


#start time of training procedure
tic = time.time()


#feed foward, back propagation to update weights and biases.
#minimize error term using method of steepest descent (MOSD)
for i in range(1, numEpochs+1):

    print("Epoch: ", i)

    #feed forward and back prop for each training data point once.
    for j in range(0, len(xTrain)):

        x1 = xTrain[j][0]; #LAC
        x2 = xTrain[j][1]; #SOW
        d_3 = xTrain[j][2]; #TACA (label)

        A_j = activity_function(x1, x2, w1j, w2j, bias_j)
        y_j = sigmoid_activation_function(A_j);
        #print("before update: ", i, " x1: ", x1, " x2: ", x2, " d_3: ", d_3, " y_j: ", y_j, " w1j: ", w1j, " w2j: ", w2j, " bias_j: ", bias_j)
        w1j = w1j + eta*(d_j - y_j)*y_j*(1-y_j)*x1 #MOSD on weight 1
        w2j = w2j + eta*(d_j - y_j)*y_j*(1-y_j)*x2 #MOSD on weight 2
        bias_j = bias_j + eta*y_j*(1 - y_j)*(d_j - y_j) #MOSD on bias
        #print("after update: ", i, " x1: ", x1, " x2: ", x2, " d_3: ", d_3, " y_j: ", y_j, " w1j: ", w1j, " w2j: ", w2j, " bias_j: ", bias_j)

#stop time of training procedure
toc = time.time()

print("Training Time (s): ", toc-tic)
print("Model Weights & Biases")
print("w1j: ", w1j, " w2j: ", w2j, " bias_j: ", bias_j)

totalError = 0;

#network has been trained, now feed the test data through the model.
for i in range(0, len(xTest)):
    x1 = xTest[i][0]; #LAC
    x2 = xTest[i][1]; #SOW
    d_3 = xTest[i][2]; #TACA (label)

    A_j = activity_function(x1, x2, w1j, w2j, bias_j)
    y_j = sigmoid_activation_function(A_j);

    # output node(s) mean squared error
    E = 0.5*pow(y_j - d_3, 2)
    totalError = totalError + E


    #threshold logic
    thresh_taca = -1;
    if y_j < 0.13:
        thresh_taca = 0;
    if y_j >= 0.13:
        thresh_taca = 1;


    print("test data point: ", i, " LAC: ", x1, " SOW: ", x2, " Predicted TACA: ", y_j, " thresh_taca: ", thresh_taca, " Actual TACA: ", d_3, " E: ", E )

print("Total Error: ", totalError)
