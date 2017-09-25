# Maxout pruning

This git contains the implementation in python using caffe of the paper "Neuron Pruning for Compressing Deep Networks Using Maxout Architectures".

It provides a simple example of the size-reduction of a conv neural network using maxout units for classifying the MNIST dataset using the LeNet (method used also for face-identification on VGGdataset).

## Abstract

This paper presents an efficient and robust approach for reducing the size of deep neural networks by pruning entire neurons. It exploits maxout units for combining neurons into more complex convex functions and it makes use of a local relevance measurement that ranks neurons according to their activation on the training set for pruning them. Additionally, a parameter reduction comparison between neuron and weight pruning is shown. It will be empirically shown that the proposed neuron pruning reduces the number of parameters dramatically. The evaluation is performed on two tasks, the MNIST handwritten digit recognition and the LFW face verification, using a LeNet-5 and a VGG16 network architecture. The network size is reduced by up to 74% and 61%, respectively, without affecting the network’s performance. The main advantage of neuron pruning is its direct influence on the size of the network architecture. Furthermore, it will be shown that neuron pruning can be combined with subsequent weight pruning, reducing the size of the LeNet-5 and VGG16 up to 92% and 80% respectively.


https://link.springer.com/chapter/10.1007/978-3-319-66709-6_15

## Example

Running the main.py script, the LeNet will be trained on Mnist dataset (LMDBs provided in Dataset folder), and iteratively the number of weights is reduced by removing neurons in the fc3 (see paper for details), based on the number of input-neurons per maxout unit. THe idea is to reduced the network withouth affecting negatively the performance.


