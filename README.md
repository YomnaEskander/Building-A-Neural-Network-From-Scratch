# Building A Nueral Network From Scratch
I have decided to try implementing neural networks from scratch. Neural Network is a very important concept in AI and since platforms such as TensorFlow and Pytorch has been simplifying it and made it very high leveled, I've decided to dig deep and try implementing it from scratch following this video by Samson Zhang on youtube  https://www.youtube.com/watch?v=w8yWXqWQYmU

# Problem Statement 
The task is digit classification using the MNIST dataset.The goal is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.
What is the MNIST dataset? 
It is a very large dataset of handwritten digits, it contains 60,000 training images and 10,000 testing images. They're in the shape of (28,28,1) which shows a low resolution and can also be said they're 28x28 pixels. Each pixel has a value between 0 and 255 which makes them gray scale. 

![image](https://github.com/YomnaEskander/Building-A-Nueral-Network-From-Scratch/assets/136505151/a7da45ec-6c5a-4efe-af2e-a13447527f3f)

The neural network will be trained on the <b>MNIST dataset</b> and will be able to tell what is the digit written. 

# The Math 

Each picture is 784 pixels as I've mentioned above which means, they could be dealt with as matrices. To make calculations, we will use the picture as matrices that are 28x28. So we have m training images represented as one matrix. Each row will be an example so will have 784 columns long, but then we will transpose this matrix and this will result in having each exmaple as a column with 784 rows. So the first column will be the first example wih 784 rows and m columns corresponding to m training examples. 

The Neural Network will be very simple with only two layers with the 0th layer (input layer) having 784 nodes, each node will have a pixel. the 1st layer will be a hidden layer with 10 units and the 2nd layer will have also 10 units.

<h4>This is the network we will be working with</h4> 

![image](https://github.com/YomnaEskander/Building-A-Nueral-Network-From-Scratch/assets/136505151/1f5b8f4f-44d9-425e-96c1-c732e28e89fc)


Three parts to training the network, forwardpropagation (or as i like to call it the forward pass), 

 <h3> Forward Propagation </h3>
it's taking an image and running it through the network and then compute the output. The math behind how neural networks work is basically matrix multiplication, therefore there will be a couple of matrices we should know. First is the A0 matrix which is a matrix containing the inputs, it's the input layer with no processing at all. then we have the Z1, which is the unactivated first layer. it's the dot product between W1 the weights and A0 the inputs + the bias. This will result in the Z matrix which will then be taken into an activation function or let's say, we will activate the Z matrix.

![image](https://github.com/YomnaEskander/Building-A-Nueral-Network-From-Scratch/assets/136505151/c2b34751-aa9a-4ce8-b07a-d0deec98609d)


<b>What happens if we did NOT apply the activation function and why activation functions are important?</b> if we didn't apply activation functions the nodes will just be the result of linear combinations of the nodes before + bias and the next nodes (next layer) will also be just a linear combination of nodes from the first layer which is just the linear combination of the input layer so this will result in only having linear combination so our neural network will be just a linear transformations to the inputs so it won't be capable of learning complicated ways to solve complicated problems, activation functions introduce non-linearity to the model. 

There are a bunch of activation functions like tanh, relu, sigmoid and softmax. We will use relu. 
![image](https://github.com/YomnaEskander/Building-A-Nueral-Network-From-Scratch/assets/136505151/723cf32b-fa4e-432d-84ca-cbf841f6a621)

After applying the relu function on the Z1 matrix we will get the A1 matrix which is again, the relu function applied to every value in Z1. this will also happen when we go from layer 1 to layer 2, to get A2. But first we will get Z2 which is also the unactivated values and then put them in the activation function SOFTMAX and get A2. Z2 will be calculated from the 2nd weight matrix for the second layer multipled by A1 + b2. We used softmax because it's better for the output layer. 
![image](https://github.com/YomnaEskander/Building-A-Nueral-Network-From-Scratch/assets/136505151/c5aa4492-0e77-444d-add2-e8ff19b7e72f)

The output layer needs to output probabilities and softmax does this. Each of the ten nodes corresponds to each digit so each node must have a probability to tell how likely the output is a certain digit. The softmax does this, it takes each node and does a calculation to get it's probability. Each output after the softmax will give a value between 0 and 1. 

<h5>The forward pass is not enough because we need to optimize the weights and biases so that the predictions are better and more accurate so machine learning gives us the opprotunity to teach our model these weights and biases so that it reaches the optimial one which will be doen using backprobagation.</h5> 

<h3>Backprobagation</h3>



side note: in the video, the instructor explained that a layer is called a layer when it has parameters and that's why he called the input layer 0th layer because it doesn't really have any parameters so it's not really a layer. 
This project is an implementation to this video https://www.youtube.com/watch?v=w8yWXqWQYmU by Samson Zhang.
