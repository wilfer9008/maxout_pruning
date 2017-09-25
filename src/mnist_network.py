'''
Created on Dec 5, 2016

@author: Fernando Moya Rueda
@email: fernando.moya@tu-dortmund.de
'''

import io
import numpy as np
import caffe
from caffe import layers as L
from caffe import params as P

class mnist(object):
    '''
    classdocs
    '''


    def __init__(self, lmdb_train_path, lmdb_test_path, prototxt_path):
        '''
        Constructor
        '''
        
        self.lmdb_train_path = lmdb_train_path
        self.lmdb_test_path = lmdb_test_path
        self.prototxt_path = prototxt_path
        


    def get_mnist_dnn(self, img_lmdb_path, num_classes, batch_size, layers_size=[512, 512, 512], use_maxout=False, train = True, k = 2):
        '''
        Returns a simple four layer dnn
        cf. baseline system
        '''
        n = caffe.NetSpec()
        
        n.images, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=img_lmdb_path, 
                                   prefetch=20, ntop=2) 

        # 
        # Traditional part of fc layers
        #
        n.layer_1 = L.InnerProduct(n.images, num_output=layers_size[0], weight_filler=dict(type='xavier'))
        n.relu1 = L.ReLU(n.layer_1, in_place=True)
        n.layer_2 = L.InnerProduct(n.relu1, num_output=layers_size[1], weight_filler=dict(type='xavier'))
        n.relu2 = L.ReLU(n.layer_2, in_place=True)
        n.fc_3 = L.InnerProduct(n.relu2, num_output=layers_size[2], weight_filler=dict(type='xavier'))
        n.relu3 = L.ReLU(n.fc_3, in_place=True)
        
        if use_maxout:
            if train:
                n.droppremax = L.Dropout(n.relu3, dropout_ratio=0.5, in_place=False, include=dict(phase=caffe.TRAIN))
                n.premax = L.Reshape(n.droppremax, shape=dict(dim=[batch_size, layers_size[2]/k,k,1]))
                n.maxout = L.Pooling(n.premax, pool=P.Pooling.MAX, global_pooling=True)
                n.fc_classes = L.InnerProduct(n.maxout, num_output=num_classes, 
                                 weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
            else:
                n.premax = L.Reshape(n.fc3, shape=dict(dim=[batch_size,layers_size[2]/k,k,1]))
                n.maxout = L.Pooling(n.premax, pool=P.Pooling.MAX, global_pooling=True)
                n.fc_classes = L.InnerProduct(n.maxout, num_output=num_classes, 
                                 weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
        else:
            n.fc_classes = L.InnerProduct(n.relu3, num_output=num_classes, 
                                 weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
        n.class_proba = L.Softmax(n.fc_classes)
        n.loss = L.SoftmaxWithLoss(n.fc_classes, n.label)
        n.acc = L.Accuracy(n.fc_classes, n.label)
        
        return n.to_proto()
    
    
        
        
    
    def get_mnist_lenet(self, img_lmdb_path, num_classes, batch_size, layers_size=[20, 50, 512], use_maxout=False, train = True, k = 2):
        '''
        Returns the lenet
        cf. baseline system
        '''
        n = caffe.NetSpec()
        
        n.images, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=img_lmdb_path, 
                                   transform_param=dict(scale=1./255), prefetch=20, ntop=2) 

        # 
        # Traditional part of fc layers
        #
        n.conv1 = L.Convolution(n.images, kernel_size=5, num_output=layers_size[0], weight_filler=dict(type='xavier'),
                                bias_filler=dict(type='constant'))
        n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
        n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=layers_size[1], weight_filler=dict(type='xavier'),
                                bias_filler=dict(type='constant'))
        n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
        n.fc_3 = L.InnerProduct(n.pool2, num_output=layers_size[2], weight_filler=dict(type='xavier'))
        n.relu3 = L.ReLU(n.fc_3, in_place=True)
        if use_maxout:
            if train:
                n.droppremax = L.Dropout(n.relu3, dropout_ratio=0.5, in_place=False, include=dict(phase=caffe.TRAIN))
                n.premax = L.Reshape(n.droppremax, shape=dict(dim=[batch_size,layers_size[2]/k,k,1]))
                n.maxout = L.Pooling(n.premax, pool=P.Pooling.MAX, global_pooling=True)
                n.fc_classes = L.InnerProduct(n.maxout, num_output=num_classes, 
                                 weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
            else:
                n.premax = L.Reshape(n.relu3, shape=dict(dim=[batch_size,layers_size[2]/k,k,1]))
                n.maxout = L.Pooling(n.premax, pool=P.Pooling.MAX, global_pooling=True)
                n.fc_classes = L.InnerProduct(n.maxout, num_output=num_classes, 
                                 weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
        else:
            n.fc_classes = L.InnerProduct(n.relu3, num_output=num_classes, param = [dict(lr_mult=1),dict(lr_mult=2)],
                                          weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
        n.class_proba = L.Softmax(n.fc_classes)
        n.loss = L.SoftmaxWithLoss(n.fc_classes, n.label)
        n.acc = L.Accuracy(n.fc_classes, n.label)

        
        return n.to_proto()
    

    def create_solver(self, idx, size_fc, niter, learning_rate, step_size, momentum, batch_size, old_solver = None, k=2, use_maxout=True):
        
        train_dnn = self.get_proto(self.lmdb_train_path, cnn_type='lenet', batch_size=batch_size, fcSize=size_fc, train_n=True, k=k, use_maxout=use_maxout)
        test_dnn = self.get_proto(self.lmdb_test_path, cnn_type='lenet', batch_size=1, fcSize=size_fc, train_n=False, k=k, use_maxout=use_maxout)
        
        cnn_architecture='mnist_maxout_' + str(idx)
        train_net_proto_path = self.prototxt_path + 'mnist_train_' + str(idx) + '.prototxt'
        test_net_proto_path = self.prototxt_path + 'mnist_test_' + str(idx) + '.prototxt'
        solver_proto_path = self.prototxt_path + 'mnist_solver_' + str(idx) + '.prototxt'
            
        with open(train_net_proto_path, 'w') as proto_file:
                proto_file.write('#%s train \n' % (cnn_architecture))
                proto_file.write(str(train_dnn))
        
        with open(test_net_proto_path, 'w') as proto_file:
                proto_file.write('#%s test \n' % (cnn_architecture))
                proto_file.write(str(test_dnn))
       
        #
        # Create solver
        #     
        solver_list = []
        solver_list.append('train_net: "%s"' % (train_net_proto_path))
        solver_list.append('test_net: "%s"' % (test_net_proto_path))
        solver_list.append('test_iter: %d' % (0))   # Set caffe library testing to zero! Will be evaluated in python!
        solver_list.append('test_interval: %d' % (niter+1))
        solver_list.append('base_lr: %f' % (learning_rate))
        solver_list.append('lr_policy: "inv"')
        solver_list.append('gamma: 0.0001')
        solver_list.append('power: 0.75')
        #solver_list.append('stepsize: %d' % (step_size))
        solver_list.append('display: %d' % (niter+1))
        solver_list.append('iter_size: %d' % (1))
        solver_list.append('max_iter: %d' % (niter))
        solver_list.append('momentum: 0.9')
        #solver_list.append('momentum: %d' % (momentum))
        solver_list.append('weight_decay: 0.0005')
        solver_list.append('average_loss: %d' % (niter+1))
        solver_list.append('snapshot: 0')
        solver_list.append('solver_mode: GPU')
        self.write_list(file_path=solver_proto_path, line_list=solver_list)
        
        #time.sleep(2)
        
        solver = caffe.get_solver(solver_proto_path)
        print 'Solver loaded'
        
        return solver
    

    def get_proto(self, lmdb_path, cnn_type='dnn', batch_size=64, fcSize = [512, 512, 128], train_n=True, k=2, use_maxout=True):
        
        if cnn_type == 'dnn':
            return str(self.get_mnist_dnn(lmdb_path, 10, batch_size, layers_size = fcSize, use_maxout=use_maxout, train=train_n, k=k))
        if cnn_type == 'lenet':
            return str(self.get_mnist_lenet(lmdb_path, 10, batch_size, layers_size = fcSize, use_maxout=use_maxout, train=train_n, k=k))
        
        

    def write_list(self, file_path, line_list, encoding='ascii'):
        '''
        Writes a list into the given file object
        
        file_path: the file path that will be written to
        line_list: the list of strings that will be written
        '''        
        with io.open(file_path, 'w', encoding=encoding) as f:
            for l in line_list:
                f.write(unicode(l) + '\n')
        