import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils import *
np.random.seed(1)

#Define the parameters of the plot or image!
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#Load the data
print('\nLoading the data...\n')
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
print('Data loaded successfully!\n\n')

#Explore your dataset
print('Exploring your dataset!')
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_x_orig shape: " + str(train_x_orig.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x_orig shape: " + str(test_x_orig.shape))
print("test_y shape: " + str(test_y.shape))

print('---------------------------------------------------------------------\n\n')

#Reset and standardize the data
print('Reshaping and standardizing Your data!\n')
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
print('After Reshaping and Standardizing: \n')
print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))

print('----------------------------------------------------------------------\n\n')

#Constants
layer_dims = [12288, 20, 7, 5, 1]
learning_rate = 0.0075

print('Performing and running your model!!\n\n')

def L_layer_model(X, Y, layer_dims, num_iterations = 3000, learning_rate = learning_rate, print_cost = False):
    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layer_dims)

    for i in range(0, num_iterations):
        #Forward propagation
        AL, caches = L_model_forward(X, parameters)
        #Cost Computing
        cost = compute_cost(AL, Y)
        #Back propagation
        grads = L_model_backward(AL, Y, caches)
        #Updating Parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        #print cost every 100 training examples
        if print_cost and i%100 == 0:
            print('Cost after iteration %i: %f' %(i, cost))
        if print_cost and i%100 == 0:
            costs.append(cost)

    print('-------------------------------------------------------------\n')
    pred_train = predict(train_x, train_y, parameters)
    print(pred_train)
    print('-------------------------------------------------------------\n')
    pred_test = predict(test_x, test_y, parameters)
    print(pred_test)
    print('-------------------------------------------------------------\n')

    # Plot the Cost!
    print('\n\nPlotting The Cost!!\n')
    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per 100)')
    plt.title('Learning_rate = ' + str(learning_rate))
    plt.show()

    return parameters


# CallingThe functions!!

parameters = L_layer_model(train_x, train_y, layer_dims, num_iterations = 2500, print_cost = True)
print('-------------------------------------------------------------\n')

print('\nEnd of the Model\n\n BYE!!')

#For checking prediction of your own image (RUN THIS IN JUPYTER NOTEBOOK!! [use "%matplotlib inline"])
## START CODE HERE ##
my_image = "my_image.jpg" # change this to the name of your image file
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
## END CODE HERE ##

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_image = my_image/255.
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")