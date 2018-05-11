# Convolutional-Neural-Networks-Step-by-Step
Let's implement convolutional (CONV) and pooling (POOL) layers in numpy, including both forward propagation.

Notation:

Superscript  [l]  denotes an object of the  lthlth  layer.
Example:  a[4]  is the  4th  layer activation.  W[5]  and  b[5]  are the  5th  layer parameters.
Superscript  (i)  denotes an object from the  ith  example.
Example:  x(i)  is the  ith  training example input.
Lowerscript  i  denotes the  ith  entry of a vector.
Example:  ai[l]  denotes the  ith  entry of the activations in layer  l , assuming this is a fully connected (FC) layer.
nH ,  nW  and nC  denote respectively the height, width and number of channels of a given layer. If you want to reference a specific layer  l , you can also write  nH[l] ,  nnW[l] ,  nC[l] .
nHprev ,  nWprev  and  nCprev  denote respectively the height, width and number of channels of the previous layer. If referencing a specific layer  l , this could also be denoted  nH[l−1] ,  nW[l−1] ,  nC[l−1].

We assume that you are already familiar with numpy and/or have completed the previous courses of the specialization. Let's get started!

1 - Packages
Let's first import all the packages that you will need during this assignment.

numpy is the fundamental package for scientific computing with Python.
matplotlib is a library to plot graphs in Python.
np.random.seed(1) is used to keep all the random function calls consistent. It will help us grade your work.

(see cnn.py)

2 - Outline of the Assignment
You will be implementing the building blocks of a convolutional neural network! Each function you will implement will have detailed instructions that will walk you through the steps needed:

Convolution functions, including:
Zero Padding
Convolve window
Convolution forward
Convolution backward (optional)
Pooling functions, including:
Pooling forward
Create mask
Distribute value
Pooling backward (optional)
This notebook will ask you to implement these functions from scratch in numpy. In the next notebook, you will use the TensorFlow equivalents of these functions to build the following model:

(refer images)
Note that for every forward function, there is its corresponding backward equivalent. Hence, at every step of your forward module you will store some parameters in a cache. These parameters are used to compute gradients during backpropagation.

3 - Convolutional Neural Networks
Although programming frameworks make convolutions easy to use, they remain one of the hardest concepts to understand in Deep Learning. A convolution layer transforms an input volume into an output volume of different size, as shown below.

(refer images)
In this part, you will build every step of the convolution layer. You will first implement two helper functions: one for zero padding and the other for computing the convolution function itself.

3.1 - Zero-Padding
Zero-padding adds zeros around the border of an image: (refer images)
Figure 1 : Zero-Padding
Image (3 channels, RGB) with a padding of 2.

The main benefits of padding are the following:

It allows you to use a CONV layer without necessarily shrinking the height and width of the volumes. This is important for building deeper networks, since otherwise the height/width would shrink as you go to deeper layers. An important special case is the "same" convolution, in which the height/width is exactly preserved after one layer.

It helps us keep more of the information at the border of an image. Without padding, very few values at the next layer would be affected by pixels as the edges of an image.

Exercise: Implement the following function, which pads all the images of a batch of examples X with zeros. Use np.pad. Note if you want to pad the array "a" of shape  (5,5,5,5,5)  with pad = 1 for the 2nd dimension, pad = 3 for the 4th dimension and pad = 0 for the rest, you would do:

a = np.pad(a, ((0,0), (1,1), (0,0), (3,3), (0,0)), 'constant', constant_values = (..,..))
(see cnn.py)

Expected Output:

x.shape:	(4, 3, 3, 2)
x_pad.shape:	(4, 7, 7, 2)
x[1,1]:	[[ 0.90085595 -0.68372786] [-0.12289023 -0.93576943] [-0.26788808 0.53035547]]
x_pad[1,1]:	[[ 0. 0.] [ 0. 0.] [ 0. 0.] [ 0. 0.] [ 0. 0.] [ 0. 0.] [ 0. 0.]]


3.2 - Single step of convolution
In this part, implement a single step of convolution, in which you apply the filter to a single position of the input. This will be used to build a convolutional unit, which:

Takes an input volume
Applies a filter at every position of the input
Outputs another volume (usually of different size) 

(refer images)
In a computer vision application, each value in the matrix on the left corresponds to a single pixel value, and we convolve a 3x3 filter with the image by multiplying its values element-wise with the original matrix, then summing them up and adding a bias. In this first step of the exercise, you will implement a single step of convolution, corresponding to applying a filter to just one of the positions to get a single real-valued output.

Later in this notebook, you'll apply this function to multiple positions of the input to implement the full convolutional operation.

Exercise: Implement conv_single_step().

(see cnn.py)

Expected Output:

Z	-6.99908945068
3.3 - Convolutional Neural Networks - Forward pass
In the forward pass, you will take many filters and convolve them on the input. Each 'convolution' gives you a 2D matrix output. You will then stack these outputs to get a 3D volume:

Exercise: Implement the function below to convolve the filters W on an input activation A_prev. This function takes as input A_prev, the activations output by the previous layer (for a batch of m inputs), F filters/weights denoted by W, and a bias vector denoted by b, where each filter has its own (single) bias. Finally you also have access to the hyperparameters dictionary which contains the stride and the padding.

Hint:

To select a 2x2 slice at the upper left corner of a matrix "a_prev" (shape (5,5,3)), you would do:
a_slice_prev = a_prev[0:2,0:2,:]
This will be useful when you will define a_slice_prev below, using the start/end indexes you will define.
To define a_slice you will need to first define its corners vert_start, vert_end, horiz_start and horiz_end. This figure may be helpful for you to find how each of the corner can be defined using h, w, f and s in the code below.

(refer images)

Reminder: The formulas relating the output shape of the convolution to the input shape is:
nH=⌊nHprev−f+2×pad/stride⌋+1
 
nW=⌊nWprev−f+2×pad/stride⌋+1
 
nC=number of filters used in the convolution
 
For this exercise, we won't worry about vectorization, and will just implement everything with for-loops.

(see cnn.py)

Expected Output:

Z's mean	0.0489952035289
Z[3,2,1]	[-0.61490741 -6.7439236 -2.55153897 1.75698377 3.56208902 0.53036437 5.18531798 8.75898442]
cache_conv[0][1][2][3]	[-0.20075807 0.18656139 0.41005165]


Finally, CONV layer should also contain an activation, in which case we would add the following line of code:

# Convolve the window to get back one output neuron
Z[i, h, w, c] = ...
# Apply activation
A[i, h, w, c] = activation(Z[i, h, w, c])
You don't need to do it here.

4 - Pooling layer
The pooling (POOL) layer reduces the height and width of the input. It helps reduce computation, as well as helps make feature detectors more invariant to its position in the input. The two types of pooling layers are:

Max-pooling layer: slides an ( f,f ) window over the input and stores the max value of the window in the output.

Average-pooling layer: slides an ( f,f ) window over the input and stores the average value of the window in the output.

(refer images)

These pooling layers have no parameters for backpropagation to train. However, they have hyperparameters such as the window size  ff . This specifies the height and width of the fxf window you would compute a max or average over.

4.1 - Forward Pooling
Now, you are going to implement MAX-POOL and AVG-POOL, in the same function.

Exercise: Implement the forward pass of the pooling layer. Follow the hints in the comments below.

Reminder: As there's no padding, the formulas binding the output shape of the pooling to the input shape is:
nH=⌊nHprev−f/stride⌋+1
 
nW=⌊nWprev−f/stride⌋+1
 
nC=nCprev

(see cnn.py)

Expected Output:

A =	[[[[ 1.74481176 0.86540763 1.13376944]]] [[[ 1.13162939 1.51981682 2.18557541]]]]
A =	[[[[ 0.02105773 -0.20328806 -0.40389855]]] [[[-0.22154621 0.51716526 0.48155844]]]]

Congratulations! You have now implemented the forward passes of all the layers of a convolutional network.





