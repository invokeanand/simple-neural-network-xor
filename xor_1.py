import numpy as np


def sigmoid(x):
    ##SIGMOID FUNCTION TO SQUASH THE OUTPUT BETWEEN 0 and 1
    return(1/(1 + np.exp(-x)))

def sigmoid_der(x):
    # DERIVATIVE OF SIGMOID
    return x*(1-x);



#input_x = np.array([[1,0,1],[1,1,0],[0,1,1]])

## INPUT for XOR GATES

input_x = np.array([[1,0],[1,1],[0,1],[0,0]])

## OUTPUT
actual_output=np.array([ [1],[0],[1],[0] ])

## SEED BECAUSE EACH RUN WILL GENERATE SAME RANDOM NUMBER
np.random.seed(0)

## WEIGHTS BETWEEN INPUT AND FIRST HIDDEN LAYER
weights_0= np.array(np.random.randn(2,4))

## WEIGHTS BETWEEN FIRST HIDDEN LAYER AND OUTPUT LAYER
weights_1= np.array(np.random.randn(4,1))


## TRAINING
for i in range(60000):
    
    #LAYER 0 IS INPUT
    l0 = input_x
    #LAYER 1 IS l0 * WEIGHTS_0 AND SIGMOID OF IT
    l1 = sigmoid(np.dot(l0, weights_0))
    
    #LAYER 2 IS l1 * WEIGHTS_1 and ITS SIGMOID
    l2 = sigmoid(np.dot(l1, weights_1))
    
    l2_error = (actual_output-l2)
    l2_delta = l2_error*sigmoid_der(l2)
    
    l1_error = l2_delta.dot(weights_1.T)
    l1_delta = l1_error * sigmoid_der(l1)
    
    weights_0 += np.dot(l0.T, l1_delta)
    weights_1 += np.dot(l1.T, l2_delta)
    
    print (l2_error)

print (l2)
