#using method of steepest descent on a multi-layer network (input layer, to hidden layer with two nodes, to output layer with single node)
#to map I/O pairs x1 = (x11, x12), x2 = (x21, x22), ..., to desired output values, d31, d32, ..., at node 3, respectively.

import numpy as np
np.set_printoptions(precision=4, suppress=True)
import time


# implements a linear activity function of a single node with two inputs
def activity_function(input1, input2, weight1, weight2, bias):
    # activity value
    A = input1*weight1 + input2*weight2 + bias
    return A


#implements the softmax activation function
def softmax_activation_function(A):
    return np.log(1 + np.exp(A))


#implements the sigmoid activation function
def sigmoid_activation_function(A):
    return 1/(1.0 + np.exp(-1*A))


#implements the sigmoid activation function
def activation(A):
    return 1/(1.0 + np.exp(-1*A))


#flags and configurations
update_biases = 1; #1 updates, 0 no updates.
test_model = 1; #1 test model, 0 do not test model.
training_method_1 = 1; #alternate input values from iteration to iteration
training_method_2 = 0; #alternate input values half way through total interations.
chosen_method = -1;


#data format [LAC, SOW, TACA] #train on LAC & SOW to predict TACA score of 1
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


# node 1 initial weights & bias
w11 = np.random.rand()
w21 = np.random.rand()
b1 = np.random.rand()

# node 2 initial weights & bias
w12 = np.random.rand()
w22 = np.random.rand()
b2 = np.random.rand()

# node 3 initial weights & bias
w_h13 = np.random.rand()
w_h23 = np.random.rand()
b3 = np.random.rand()

#breakpoint()

eta = 0.1 #learn rate
num_epochs= 30 #epochs (number of times the entire dataset is seen)


print("CONFIGURATIONS: ")
print("update biases: ", update_biases)
print("eta: ", eta)
print("epochs: ", num_epochs)
print("")


#start time of training procedure
tic = time.time()

#for each epoch feed foward and back prop each data point
#to update weights and biases.
#minimize error term using method of steepest descent (MOSD)
for i in range(1, num_epochs+1):  #each epoch

    print("Epoch: ", i)

    #feed forward and back prop for each training data point once.
    for j in range(0, len(xTrain)):

        x1 = xTrain[j][0]; #LAC
        x2 = xTrain[j][1]; #SOW
        d_3 = xTrain[j][2]; #TACA (label) desired output at node 3

        #print("feeding data point: ", i, " x1: ", x1, " x2: ", x2, " d_3: ", d_3)

        # node 1 feed forward (input layer to node 1)
        A_1 = activity_function(x1, x2, w11, w21, b1)
        y_1 = sigmoid_activation_function(A_1);

        # node 2 feed forward (input layer to node 2)
        A_2 = activity_function(x1, x2, w12, w22, b2)
        y_2 = sigmoid_activation_function(A_2);

        # node 3 feed forward (node 1 & nose 2 outputs, to node 3)
        A_3 = activity_function(y_1, y_2, w_h13, w_h23, b3)
        y_3 = sigmoid_activation_function(A_3);


        '''
        print("before update: ", i)
        print("y_1: ", y_1, " y_2: ", y_2, " y_3: ", y_3)
        print("A_1: ", A_1, " A_2: ", A_2, " A_3: ", A_3)
        print("w11: ", w11, " w21: ", w21, " b1: ", b1)
        print("w12: ", w12, " w22: ", w22, " b2: ", b2)
        print("w_h13: ", w_h13, " w_h23: ", w_h23, " b3: ", b3)
        bigE = 0.5*pow(y_3 - d_3, 2);
        print("output node(s) total error, E: ", bigE)
        '''


        #delta for output node 3
        delta_3 = (d_3 - y_3)*(1 - y_3)*y_3

        #delta for hidden layer node 2
        delta_2 = (1 - y_2)*y_2*delta_3*w_h23

        #delta for hidden layer node 1
        delta_1 = (1 - y_1)*y_1*delta_3*w_h13


        # update node 3 (output layer node) weights & bias
        w_h13_new = w_h13 + eta*delta_3*y_1 #MOSD on weight from node 1 to node 3
        w_h23_new = w_h23 + eta*delta_3*y_2 #MOSD on weight from node 2 to node 3
        if update_biases == 1:
            b3_new = b3 + eta*delta_3 #MOSD on bias of node 3


        # update node 2 (hidden layer node) weights & biase
        w12_new = w12 + eta*delta_2*x1 #MOSD on weight from input 1 to node 2
        w22_new = w22 + eta*delta_2*x2 #MOSD on weight from input 2 to node 2
        if update_biases == 1:
            b2_new = b2 + eta*delta_2 #MOSD on bias of node 2


        # update node 1 (hidden layer node) weights & bias
        w11_new = w11 + eta*delta_1*x1 #MOSD on weight from input 1 to node 1
        w21_new = w21 + eta*delta_1*x2 #MOSD on weight from input 2 to node 1
        if update_biases == 1:
            b1_new = b1 + eta*delta_1 #MOSD on bias of node 1


        #after updating the weights & biases using the old values, we store the new, updated values
        w_h13 = w_h13_new
        w_h23 = w_h23_new
        w12 = w12_new
        w22 = w22_new
        w11 = w11_new
        w21 = w21_new
        if update_biases == 1:
            b1 = b1_new
            b2 = b2_new
            b3 = b3_new

        '''
        print("after update: ", i)
        print("y_1: ", y_1, " y_2: ", y_2, " y_3: ", y_3)
        print("A_1: ", A_1, " A_2: ", A_2, " A_3: ", A_3)
        print("w11: ", w11, " w21: ", w21, " b1: ", b1)
        print("w12: ", w12, " w22: ", w22, " b2: ", b2)
        print("w_h13: ", w_h13, " w_h23: ", w_h23, " b3: ", b3)
        bigE = 0.5*pow(y_3 - d_3, 2)
        print("output node(s) total error, E: ", bigE)
        print("")
        '''

print("")
print("Model Weights & Biases")
print("w11: ", w11, " w21: ", w21, " b1: ", b1)
print("w12: ", w12, " w22: ", w22, " b2: ", b2)
print("w_h13: ", w_h13, " w_h23: ", w_h23, " b3: ", b3)
print("")

#stop time of training procedure
toc = time.time()
print("Training Time (s): ", toc-tic)

totalError = 0


#this is where we test the model against the testing data and look
#at the predicted TACA values.
if test_model == 1 and i == num_epochs:

    outputValues = [[0 for _ in range(4)] for _ in range(10)]

    #feed forward and back prop for each training data point once.
    for j in range(0, len(xTest)):

        x1 = xTest[j][0]; #LAC
        x2 = xTest[j][1]; #SOW
        d_3 = xTest[j][2]; #TACA (label)

        # node 1 feed forward (input layer to node 1)
        A_1 = activity_function(x1, x2, w11, w21, b1)
        y_1 = sigmoid_activation_function(A_1);

        # node 2 feed forward (input layer to node 2)
        A_2 = activity_function(x1, x2, w12, w22, b2)
        y_2 = sigmoid_activation_function(A_2);

        # node 3 feed forward (node 1 & nose 2 outputs, to node 3)
        A_3 = activity_function(y_1, y_2, w_h13, w_h23, b3)
        y_3 = sigmoid_activation_function(A_3);

        # output node(s) mean squared error
        E = 0.5*pow(y_3 - d_3, 2)

        totalError = totalError + E

        #threshold logic
        thresh_taca = -1;
        if y_3 < 0.49:
            thresh_taca = 0;
        if y_3 >= 0.49:
            thresh_taca = 1;

        outputValues[j][0] = d_3 #truth
        outputValues[j][1] = thresh_taca #thresholded
        outputValues[j][2] = y_3 #predicted
        outputValues[j][3] = E #error

        print("test data point: ", j, " LAC: ", x1, " SOW: ", x2, " Predicted TACA: ", y_3, " Actual TACA: ", d_3, " thresh_taca: ", thresh_taca, " E: ", E)

print("Total Error: ", totalError)
