'''
Created on Dec 5, 2016

@author: Fernando Moya Rueda
@email: fernando.moya@tu-dortmund.de
'''

import numpy as np
import os
import caffe

from mnist_network import mnist    



def count_act_neurons(solver_net, size_fc, sizeTrain, batchS, k):
    '''
    Returns the relevance measure of each neuron per maxout unit.
    The relevance is just the frequency a neuron becomes
    the maximum in the maxout unit.

    @param solver_net: Solver
    @param size_fc: Size of the fully connected layer
    @param sizeTrain: Number of samples in training data
    @param batchS: Batch size
    @param k: k value(see paper), number of neurons per maxout unit
    '''
    
    # At the end run over training set
    fc3_blob_count = np.zeros( (int(size_fc/k), k ))
    for item_num in range(int(sizeTrain/batchS)):
        solver_net.net.forward()
        premax_data = solver_net.net.blobs['premax'].data.copy()
        for b in range(batchS):
            fc3_data_b = premax_data[b,:,:,0]
            argM = np.argmax(fc3_data_b, axis = 1)
            for a in range(len(argM)):
                if False:#(len(set(fc3_data_b[a,:])) <= 1):
                    for neuron in range(k):
                        fc3_blob_count[a,neuron] +=1
                    continue
                else:
                    if argM[a]==0:
                        fc3_blob_count[a,0] +=1
                        continue
                    if argM[a]==1:
                        fc3_blob_count[a,1] +=1
                        continue
                    if argM[a]==2:
                        fc3_blob_count[a,2] +=1
                        continue
                    if argM[a]==3:
                        fc3_blob_count[a,3] +=1
                        continue         
    
    return fc3_blob_count



def test_net(solver_net, sizeTest):
    '''
    Test the network.
    It returns the classification accuracy

    @param solver_net: Solver
    @param sizeTest: Number of samples in the test dataset
    '''
    
    accuracy_counter = 0
    for item_num in range(sizeTest):
        solver_net.test_nets[0].forward()
        if np.argmax(solver_net.test_nets[0].blobs['class_proba'].data.copy()) == np.array(solver_net.test_nets[0].blobs['label'].data.copy(),dtype=np.int64 )[0]:
            accuracy_counter = accuracy_counter + 1
    
    return accuracy_counter/float(sizeTest)


def train(solver_net, niter, sizeTest, disp_interval, test_interval):
    '''
    Trains the network. 
    And it returns the testing accuracies (for ploting if wanted), only matters the last one

    @param solver_net: Solver
    @param niter: Number of iterations
    @param sizeTest: Number of samples in testing data
    @param disp_interval: Interval for showing the loss and acc
    @param test_interval: Testing interval
    '''
    
    loss = np.zeros(niter)
    acc = np.zeros(niter)
    accuracies = []
    
    
    for it in range(niter):
        solver_net.step(1) #Run a single SGD step # Simulate a batch size of 
        loss[it] = solver_net.net.blobs['loss'].data.copy()
        acc[it] = solver_net.net.blobs['acc'].data.copy()
        
        #Print loss and accuracies each dips_interval
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = 'Loss:', loss[it], ' Acc:', np.round(100*acc[it])
            print '%3d) %s' % (it, loss_disp)
            
        #Test the network each test_interval
        if it % test_interval == 0 or it + 1 == niter:
            partial_acc = test_net(solver_net, sizeTest)
            accuracies.append(partial_acc)
            print "Accuracy test:", partial_acc
        
    return accuracies


def neuron_pruning(solver_net, size_fc, sizeTrain, batchS, k, pruned):
    '''
    Pruns the network. It gets a network, computes the neuron relevance per maxout unit
    Creates new network and copy the weights of the neurons with the bigger relevance
    It returns a smaller network (reduction only in the layer fc3, see paper) 

    @param solver_net: Solver
    @param size_fc: Size of the fully connected layer
    @param sizeTrain: Number of samples in training data
    @param batchS: Batch size
    @param k: k value(see paper), number of neurons per maxout unit
    @param pruned: Number of pruned neurons
    '''
    
    #Computing the relevance measure.
    #It counts the frequency neurons become maximal per maxout unit
    neurons_relevance = count_act_neurons(solver_net, size_fc, sizeTrain, batchS, k) 
    print neurons_relevance.shape
    
    
    #One deletes the neurons with the smaller relevance
    relevance_sort = np.argsort(neurons_relevance, axis = 1)
    survival_neurons = relevance_sort[:,1:]
    survival_neurons = np.sort(survival_neurons, axis = 1)
    
    #Getting old weights W and b
    W = solver_net.net.params['fc_3'][0].data[...].copy()
    b = solver_net.net.params['fc_3'][1].data[...].copy()
    
    print W.shape, b.shape
    
    #Create zeroed-matrices for containing the new weights
    W_new = np.zeros((int(size_fc*(k-pruned-1)/k),W.shape[1]))
    b_new = np.zeros((int(size_fc*(k-pruned-1)/k)))
    
    print W_new.shape, b_new.shape
    
    #Copying the survival weights, keeping order
    for v in range(survival_neurons.shape[0]):
        for neuron in range((k-pruned)-1):
            W_new[v+neuron,:] = W[v*(k-pruned)+survival_neurons[v,neuron]]
            b_new[v+neuron] = b[v*(k-pruned)+survival_neurons[v,neuron]]
            

    #Getting weights from old network
    fc1_w = solver_net.net.params['conv1'][0].data[...].copy()
    fc1_b = solver_net.net.params['conv1'][1].data[...].copy()
    fc2_w = solver_net.net.params['conv2'][0].data[...].copy()
    fc2_b = solver_net.net.params['conv2'][1].data[...].copy()  
    fc_classes_w = solver_net.net.params['fc_classes'][0].data[...].copy()
    fc_classes_b = solver_net.net.params['fc_classes'][1].data[...].copy()
    
    #Creates solver and networks (train and test)
    solver_2 = mnist_network.create_solver(pruned + 1, [20, 52, size_fc*(k-pruned-1)/k], niter, learning_rate, step_size, momentum, batchS, k = (k-pruned-1))

    #Copying the new weights into the network
    solver_2.net.params['conv1'][0].data[...] = fc1_w
    solver_2.net.params['conv1'][1].data[...] = fc1_b
    solver_2.net.params['conv2'][0].data[...] = fc2_w
    solver_2.net.params['conv2'][1].data[...] = fc2_b
    solver_2.net.params['fc_3'][0].data[...] = W_new
    solver_2.net.params['fc_3'][1].data[...] = b_new
    solver_2.net.params['fc_classes'][0].data[...] = fc_classes_w
    solver_2.net.params['fc_classes'][1].data[...] = fc_classes_b
        
    return solver_2



def run_solver(niter = 10000, disp_interval=500, test_interval=12000,
               step_size=1500, learning_rate=0.01, momentum=0.9,
               batchSize = 64, size_fc=128, k = 4):

    '''
    Runs the solver, training, pruning and testing the network

    @param niter: Number of iterations
    @param disp_interval: Interval for showing the loss and acc
    @param test_interval: Testing interval
    @param learning_rate: Learning rate, default 0.0.1
    @param momentum: Momentum, default 0.9
    @param batchSize: Batch size
    @param size_fc: Number of neurons in last fully connected layer
    @param k: k value(see paper), number of neurons per maxout unit
    '''
    
    accuracies_all_nets = np.zeros(k)
    
    #Creates initial solver and networks (train and test) with k maxout units
    solver = mnist_network.create_solver(0, [20, 52, size_fc], niter, learning_rate, step_size, momentum, batchSize, k=k, use_maxout=True)
    
    #Trains initial network
    print "Training network, pruning # ", 0
    accuracies = train(solver, niter, sizeTest, disp_interval, test_interval)
    accuracies_all_nets[0] = accuracies[-1] * 100
    print "Final Accuracy ", accuracies[-1] * 100, "%\n"

    #Pruning and training network
    for pruned in range(k-1):
        print "Pruning layer"
        solver = neuron_pruning(solver, size_fc, sizeTrain, batchSize, k, pruned)
        print "Training network, pruning # ", pruned + 1 
        accuracies = train(solver, niter, sizeTest, disp_interval, test_interval)
        accuracies_all_nets[pruned + 1 ] = accuracies[-1] * 100
        print "Final Accuracy ", accuracies[-1] * 100, "%\n"
        
    print accuracies_all_nets


    # Saving the learned weights.
    print "Saving the learned weights."
    weights = os.path.join(weight_dir, 'weights_mnist_.caffemodel')
    solver.net.save(weights)  
    
    
    
    return

if __name__ == '__main__':
    caffe.set_device(0)
    caffe.set_mode_gpu()
    

    #Setting directories of the datasets, models, and weights
    model_root = '../mnist_prototxt/'
    lmdb_test = '../Dataset/mnist_test_lmdb/'
    lmdb_train = '../Dataset/mnist_train_lmdb/'
    weight_dir = '../trained_models/' 
    
    #Setting some hyperparameters
    sizeTest = 10000
    sizeTrain = 60000
    batchSize = 64
    epochs = 4
    niter = epochs * (sizeTrain // batchSize) + 1
    disp_interval = niter // 50
    test_interval = niter // 2
    step_size = niter // 4
    learning_rate = 0.01
    momentum = 0.9
    size_fc = 256
    k = 4

    if (size_fc / float(k)) % 2 != 0:
        raise AssertionError("Change size of the layer or k value! They should satisfy (size_fc / 4) mod 2 == 0")
    

    mnist_network = mnist(lmdb_train, lmdb_test, model_root)
    
    
    run_solver(niter, disp_interval, test_interval,
               step_size, learning_rate, momentum,
               batchSize, size_fc, k)
    print 'Done'