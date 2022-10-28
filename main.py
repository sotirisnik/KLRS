#!/usr/bin/env python
# coding: utf-8
"""
Example usage:

python klrs_version1.py --local_buffer rs --local_buffer_size=500 --batch_size 10 --replay_size 10 --inner_steps 5 --dataset_name mnist --agent_lr 0.01 --seed 1 --dynamic_trade_off True --use_gpu True

python klrs_version1.py --local_buffer rs --local_buffer_size=500 --batch_size 10 --replay_size 10 --inner_steps 5 --dataset_name cifar10 --agent_lr 0.01 --seed 1 --dynamic_trade_off True --nn resnet18 --use_gpu True

"""

import argparse

parser = argparse.ArgumentParser( description='Continual Learning Process' )
parser.add_argument( "--inner_steps", type=int, help='Number of epochs', default=5 )
parser.add_argument( "--dataset_name", type=str, help="Dataset to use e.x. cifar10, cifar100, mnist, fashion_mnist, emnist, miniimagenet", default='mnist' )
parser.add_argument( "--nn", type=str, help="Neural Network architecture", default='nn_default' )
parser.add_argument( "--agent_lr", type=float, help="Learning rate of agent", default=0.05 )
parser.add_argument( "--dynamic_trade_off", type=bool, help="Dynamic trade_off alpha = 1/len(seen_classes) at agent", default=True )
parser.add_argument( "--trade_off", type=float, help="trade off for batch and replay loss", default=1.0 )
parser.add_argument( "--use_gpu", type=str, help="Run of gpu[True] or cpu default [False]", default="False" )
parser.add_argument( "--local_buffer", type=str, help="Local buffer for agent i.e. nobuff, rs", default="nobuff" )
parser.add_argument( "--local_buffer_size", type=int, help="Size of local buffer for clienbuff i.e. 500", default=500 )
parser.add_argument( "--replay_size", type=int, help="How many datapoints do we sample per call", default=10 )
parser.add_argument( "--replay_method", type=str, help="uniform or weighted", default="weighted" )
parser.add_argument( "--batch_size", type=int, help="How many datapoints do we receive per call from stream", default=10 )
parser.add_argument( "--freeze_top", type=bool, help="freeze top layers of resnet18", default=False )
parser.add_argument( "--pi", type=float, help="Trade off follow the stream(>0.5) or follow the uniform", default=0.5 )
parser.add_argument( "--test_scenario", type=str, help="balanced of follow_the_stream", default="balanced" )
parser.add_argument( "--kl_technique", type=str, help="kl technique", default="default" )
parser.add_argument( "--kl_alpha", type=float, help="kl alpha", default=0.5 )
parser.add_argument( "--seed", type=int, help="seed", default=1 )

args = parser.parse_args()

inner_steps = args.inner_steps
dataset_name = args.dataset_name
nn_architecture = args.nn
agent_learning_rate = args.agent_lr
dynamic_trade_off = args.dynamic_trade_off
trade_off = args.trade_off
use_gpu = args.use_gpu
local_buffer = args.local_buffer
local_buffer_size = args.local_buffer_size
replay_size = args.replay_size
replay_method = args.replay_method
batch_size = args.batch_size
freeze_top = args.freeze_top
pi = args.pi
kl_technique = args.kl_technique
kl_alpha = args.kl_alpha
test_scenario = args.test_scenario

seed = args.seed

if local_buffer not in ["nobuff","random","rs","wrs","cbrs","klrs"]:
    print( "Sorry wrong options for local buffer" )
    exit(0)

if replay_method not in ['uniform','weighted']:
    print( "Sorry wrong options for replay method" )
    exit(0)

if nn_architecture not in ['nn_default','resnet18',]:
    print( "Sorry wrong options for nn" )
    exit(0)

if test_scenario not in ['balanced','follow_the_stream']:
    print( "Sorrt wrong options for test scenario" )
    exit(0)

if kl_technique not in ['default','dynamic_pi','target']:
    print( "Sorry wrong options for kl_technique" )
    exit(0)

#import libraries
import pickle
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if use_gpu == "False":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import torch
from torch.nn import Module, Linear, ReLU
import torchvision.models as models
from model_transfer import transfer_params_to_jax
import jax.numpy as jnp
import numpy as np
import jax
import haiku as hk
import optax
import os
print( "Jax devices" )
print( jax.devices() )
from jax import grad
from jax import random
from jax.example_libraries import optimizers
#from jax.experimental import optix
from jax.example_libraries.optimizers import clip_grads
import tensorflow as tf
print( tf.__version__ )
tf.config.experimental.set_visible_devices([], "GPU")
#tf.compat.v1.enable_eager_execution()
tf.keras.backend.set_floatx('float64')
tf.executing_eagerly()
from copy import deepcopy
from jax import vmap # for auto-vectorizing functions
from functools import partial # for use with vmap
from jax import jit # for compiling functions for speedup
#from jax.experimental import stax # neural network library
#jax.experimental.stax is deprecated, import jax.example_libraries.stax instead
from jax.example_libraries import stax
from jax.example_libraries.stax import Conv, Dense, MaxPool, Relu, LeakyRelu, Flatten, Identity, LogSoftmax, BatchNorm # neural network layers
import matplotlib.pyplot as plt # visualization
from jax import jit # for compiling functions for speedup
from jax.tree_util import tree_multimap
import tensorflow_datasets as tfds
from jax.example_libraries.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                   FanOut, Flatten, GeneralConv, Identity,
                                   MaxPool, Relu, LogSoftmax)
import time
import matplotlib.pyplot as plt
import functools
from sortedcontainers import SortedList
import numpy as onp
from buffers import ReservoirSampling, Random, WRS, CBRS, KLRS
#end of import libraries

torch.manual_seed(seed)
tf.random.set_seed(seed)

print( "Settings:")
print( "\tInner steps:", inner_steps )
print( "\tDataset_name:", dataset_name )
print( "\tArchitecture", nn_architecture )
print( "\tLearning rate", agent_learning_rate )
print( "\tDynamic trade off", dynamic_trade_off )
print( "\tTrade off", trade_off )
print( "\tUse Gpu", use_gpu )
print( "\tLocal buffer", local_buffer )
print( "\tLocal buffer size", local_buffer_size )
print( "\tKL pi", pi )
print( "\tKL technique", kl_technique )
print( "\tKL alpha", kl_alpha )
print( "\tSeed", seed )

def TryToCreate( folder_struct ):
    while True:
        try:
            os.mkdir(folder_struct)
        except:
            pass
        if os.path.exists( folder_struct ) == True:
            break
    print( "Created folder", folder_struct )

folder_struct = "../out_cl"
if os.path.exists( folder_struct ) == False:
    TryToCreate( folder_struct )

folder_struct = "../out_cl/results"
if os.path.exists(folder_struct) == False:
    TryToCreate( folder_struct )
    
folder_struct = "../out_cl/results/" + dataset_name
if os.path.exists(folder_struct) == False:
    TryToCreate( folder_struct )
    
#create learning rate folder
folder_struct += "/" + str(agent_learning_rate)
if os.path.exists(folder_struct) == False:
    TryToCreate( folder_struct )
#create architecture folder
folder_struct += "/" + str(nn_architecture)
if os.path.exists(folder_struct) == False:
    TryToCreate( folder_struct )
folder_struct += "/"

print( "Results will be saved at")
print( folder_struct )
#exit(0)

if dataset_name in ['mnist','cifar10','fashion_mnist']:
    K = 10
elif dataset_name == 'cifar100':
    K = 100
elif dataset_name == 'emnist':
    K = 62
else:
    print( "Sorry not recognizable dataset")
    exit(0)

if dataset_name == 'emnist':
    emnist_ds = tfds.load( "emnist", as_supervised=True, batch_size=-1 )
elif dataset_name == 'mnist':
    mnist_ds = tfds.load( "mnist", as_supervised=True, batch_size=-1 )
elif dataset_name == 'fashion_mnist':
    fashion_mnist_ds = tfds.load( "fashion_mnist", as_supervised=True, batch_size=-1 )
elif dataset_name == 'cifar10':
    cifar10_ds = tfds.load( "cifar10", as_supervised=True, batch_size=-1 )
elif dataset_name == 'cifar100':
    cifar100_ds = tfds.load("cifar100", as_supervised=True,  batch_size=-1 )

#split categories into lists
def split_ds_into_classes( ds, key='train', num_classes=10 ):
    images_per_class, labels_per_class = [], []
    for i in range(num_classes):
        mask = ds[ key ][1].numpy() == i
        images_per_class.append( ds[ key ][0][ mask ] )
        labels_per_class.append( ds[ key ][1][ mask ] )
    return images_per_class, labels_per_class
    
def split_into_train_valid( split_rng, ipc_train, lpc_train ):
    
    images_train_split, labels_train_split, images_valid_split, labels_valid_split = [], [], [], []

    for idx in range( len(ipc_train) ):#L[.] is tf.Tensor
        perm_seq = split_rng.permutation( ipc_train[ idx ].shape[0] ).reshape( (-1,1) )#random permutation
        ipc_train[idx] = ipc_train[idx].numpy()#from tensor to numpy
        lpc_train[idx] = lpc_train[idx].numpy()#from list to numpy
        bound = int( len(perm_seq) * 0.9 )#90% train - 10% valid
        images_train_split.append( tf.Variable( ipc_train[idx][ perm_seq[ :bound ] ].squeeze(axis=1) ) )#data for train
        labels_train_split.append( lpc_train[idx][ perm_seq[ :bound ] ].squeeze(axis=1) )#data for train
        images_valid_split.append( tf.Variable( ipc_train[idx][ perm_seq[ bound: ] ].squeeze(axis=1) ) )#data for valid
        labels_valid_split.append( lpc_train[idx][ perm_seq[ bound: ] ].squeeze(axis=1) )#data for valid
    
    return images_train_split, labels_train_split, images_valid_split, labels_valid_split

split_rng = np.random.default_rng(seed)

if dataset_name == 'mnist':
    ipc_train, lpc_train = split_ds_into_classes( mnist_ds, 'train' )
    ipc_test, lpc_test = split_ds_into_classes( mnist_ds, 'test' )
    ipc_train, lpc_train, ipc_valid, lpc_valid = split_into_train_valid( split_rng, ipc_train, lpc_train )
elif dataset_name == 'cifar10':
    ipc_train, lpc_train = split_ds_into_classes( cifar10_ds, 'train' )
    ipc_test, lpc_test = split_ds_into_classes( cifar10_ds, 'test' )
    ipc_train, lpc_train, ipc_valid, lpc_valid = split_into_train_valid( split_rng, ipc_train, lpc_train )
elif dataset_name == 'cifar100':
    ipc_train, lpc_train = split_ds_into_classes( cifar100_ds, 'train', num_classes=100 )
    ipc_test, lpc_test = split_ds_into_classes( cifar100_ds, 'test', num_classes=100 )
    ipc_train, lpc_train, ipc_valid, lpc_valid = split_into_train_valid( split_rng, ipc_train, lpc_train )
elif dataset_name == 'fashion_mnist':
    ipc_train, lpc_train = split_ds_into_classes( fashion_mnist_ds, 'train' )
    ipc_test, lpc_test = split_ds_into_classes( fashion_mnist_ds, 'test' )
    ipc_train, lpc_train, ipc_valid, lpc_valid = split_into_train_valid( split_rng, ipc_train, lpc_train )
elif dataset_name == 'emnist':
    ipc_train, lpc_train = split_ds_into_classes( emnist_ds, 'train', num_classes=62 )
    ipc_test, lpc_test = split_ds_into_classes( emnist_ds, 'test', num_classes=62 )
    ipc_train, lpc_train, ipc_valid, lpc_valid = split_into_train_valid( split_rng, ipc_train, lpc_train )

#create imbalance
def generate_imbalanced_factors(step,num_classes, imbalance_rng):
    r = np.array( [ 10**(-i*step) for i in range(5) ] )#retention factors
    r = r.repeat( np.ceil(num_classes/5) )
    perm = imbalance_rng.permutation( len(r) )
    r = r[perm][ :num_classes ]
    return r

def apply_imbalanced_factors( ipc, lpc, factors ):
    for i in range( len(ipc) ):
        up_to = int( ipc[i].shape[0] * factors[i] )
        ipc[i] = deepcopy( ipc[i][ : up_to ] )
        lpc[i] = deepcopy( lpc[i][ : up_to ] )
    return ipc, lpc

def apply_permutation( ipc, lpc, perm ):
    new_ipc, new_lpc = [], []
    new_ipc = [ ipc[i] for i in perm ]
    new_lpc = [ lpc[i] for i in perm ]
    return new_ipc, new_lpc

imbalance_rng = np.random.default_rng(seed)
factors = generate_imbalanced_factors( step=0.5, num_classes=K, imbalance_rng=imbalance_rng )

perm_rng = np.random.default_rng(seed)

perm1 = perm_rng.permutation(K)
perm2 = perm_rng.permutation(K)

#permute labels
ipc_train, lpc_train = apply_permutation( ipc_train, lpc_train, perm1 )
ipc_test, lpc_test = apply_permutation( ipc_test, lpc_test, perm1 )
ipc_valid, lpc_valid = apply_permutation( ipc_valid, lpc_valid, perm1 )
#permute factors
factors = [ factors[i] for i in perm2 ]

print( 'Original Stream counts', [ i.shape[0] for i in ipc_train ] )

ipc_train, lpc_train = apply_imbalanced_factors( ipc_train, lpc_train, factors )#Apply imbalance factors on train-dataset only
print( 'Imbalanced Stream counts', [ i.shape[0] for i in ipc_train ] )
print( 'Imbalanced Stream labels', [ i[0] for i in lpc_train ] )

print( 'Test counts', [ i.shape[0] for i in ipc_test ] )
if test_scenario == 'follow_the_stream':
    ipc_test, lpc_test = apply_imbalanced_factors( ipc_test, lpc_test, factors )
    print( 'Imbalanced Test counts', [ i.shape[0] for i in ipc_test ] )
#ipc_valid, lpc_valid = apply_imbalanced_factors( ipc_valid, lpc_valid, factors )

###end of imbalance

def random_number( a, b, rng ):
    return int( (b-a) * rng.random() + a )

def random_real( a, b, rng ):
    return float( (b-a) * rng.random() + a )

def freeze_top_layers( agent_grad ):
    trainable= dict( )
    for i in agent_grad:
        trainable[i] = dict()
        for j in agent_grad[i]:
            trainable[i][j] = 0
    trainable['res_net18/~/logits']['b'] = 1
    trainable['res_net18/~/logits']['w'] = 1
    trainable['res_net18/~/logits']
    inner_fn = lambda state, trainable_params: ( state if trainable_params==1 else state * 0 )
    return tree_multimap(inner_fn, agent_grad, trainable )

def update_rule1(param, buffer_grad, replay_grad ):#, alpha, lr):
  return param - agent_learning_rate * ( (trade_off) * buffer_grad + (1-trade_off)* replay_grad )

def update_rule2(param, buffer_grad ):#, alpha, lr):
    return param - agent_learning_rate * trade_off *buffer_grad 

def ToOneHot( labels, num_classes ):
    vectors = jnp.eye( num_classes )
    lbl = jnp.array( [ vectors[ i ] for i in labels ] )
    return lbl

class DataCollector:
    
    def __init__(self, num_classes=10):
        self.train_data, self.train_labels = [], []
        self.test_data, self.test_labels = [], []
        self.valid_data, self.valid_labels = [], []
    def AddSamples( self, data, cat, mode='train' ):
        if mode == 'train':
            self.train_data.append( data )
            self.train_labels.append( cat )#[cat] * len(data) )
        elif mode == 'test':
            self.test_data.append( data )
            self.test_labels.append( cat )#[cat] * len(data) )
        elif mode == 'valid':
            self.valid_data.append( data )
            self.valid_labels.append( cat )# [cat] * len(data) )
        else:
            print( "wrong input" )
            exit(0)

def normalize( images, dataset_name ):
    if dataset_name in [ 'mnist', 'fashion_mnist', 'emnist' ]:
        #print( 'ok' )
        images = images / 255.0
        return images
    elif dataset_name in ['cifar10','cifar100']:
        if nn_architecture == 'resnet18':
            images = jax.image.resize( images, shape=(images.shape[0],224,224,3), method="bilinear", antialias=False )
        images = (images / 255.0)
        return images
    else:
        print( "Wrong dataset_name", dataset_name )
        exit(0)

class DataStream:
    
    def __init__(self,dataset_name):
        self.train_stream_images = []
        self.train_stream_labels = []
        self.test_data, self.test_labels = [], []
        self.valid_data, self.valid_labels = [], []
        self.cur_pos = 0
        self.stream_completed = False
            
    def build_stream(self, obj_arg : DataCollector, num_classes ):
        
        obj = deepcopy( obj_arg )
        
        for i in range( len(obj.train_data) ):
            self.train_stream_images.extend( deepcopy( obj.train_data[i] ) )
            self.train_stream_labels.extend( deepcopy( obj.train_labels[i] ) )
    
        self.train_stream_images = jnp.array( self.train_stream_images )
        self.train_labels_labels = jnp.array( self.train_stream_labels )
        
        #insert test/valid data and get labels
        for i in range( len(obj.test_data) ):
            self.test_data.append( jnp.array( obj.test_data[i] ) )
            self.test_labels.append( ToOneHot( jnp.array( obj.test_labels[i] ), num_classes ) )
        
        for i in range( len(obj.valid_data) ):
            self.valid_data.append( jnp.array( obj.valid_data[i] ) )
            self.valid_labels.append( ToOneHot( jnp.array( obj.valid_labels[i] ), num_classes ) )
        
    def GetNextBatch( self, batch_size=128 ):
        lo = self.cur_pos
        hi = self.cur_pos + batch_size
        
        if self.stream_completed:
            return ( False, False )
        
        if hi >= self.train_stream_images.shape[0]:
            hi = self.train_stream_images.shape[0]
            self.stream_completed = True
        else:
            self.cur_pos += batch_size
        
        return ( self.train_stream_images[lo:hi], self.train_stream_labels[lo:hi] )
    
    def hasNextBatch( self ):
        return self.stream_completed != True
    
    def ResetStream( self ):
        self.cur_pos = 0
        self.stream_completed = False
    
    def GetTestData( self ):
        return ( self.test_data, self.test_labels )

    def GetTestDataLimitClass( self, categories ):
        subset_x, subset_y = [], []
        for i, j in zip( self.test_data, self.test_labels ):
            if j[0].argmax().item() in categories:
                subset_x.append( i )
                subset_y.append( j )
        return ( subset_x, subset_y )

    def GetValidData( self ):
        return ( self.valid_data, self.valid_labels )

def ViewBatch( images, labels ):
    bcnt = 0
    plt.figure()
    for i, j in zip( images, labels ):
        bcnt += 1
        plt.subplot( 6, 24, bcnt )
        plt.axis('off')
        plt.title( j )
        plt.imshow( i.reshape( (28,28) ), cmap='gray' )
    plt.show()

rng = np.random.default_rng(seed)

agent_collector = DataCollector()
agent_stream = DataStream(dataset_name)

split_rng = np.random.default_rng(seed)

def FillDataCollect( ipc, lpc, mode='train' ):
    for i in range( len(ipc) ):#iterate each class
        current_data = ipc[i].numpy()#images per class
        current_labels = lpc[i]#.numpy()
        agent_collector.AddSamples( data=current_data, cat=current_labels, mode=mode )
        
#Fill categories for agent
FillDataCollect( ipc_train, lpc_train, 'train' )
FillDataCollect( ipc_test, lpc_test, 'test' )
FillDataCollect( ipc_valid, lpc_valid, 'valid' )

#Build training stream
agent_stream.build_stream( deepcopy(agent_collector), K )
        
class Agent:

    def __init__(self, seed, K, stream : DataCollector=None, dataset_name='mnist', buffer_size=500, buffer_type="rs", pi = 0.5, kl_technique="default", kl_alpha=0.5):
        self.K = K
        self.stream = stream#training stream and test/valid data
        self.stream.ResetStream()
        
        self.seen_categories = {}

        self.nn_rng = random.PRNGKey( seed )
        self.Lista = []
        channels = 1
        img_w, img_h = 28, 28
        
        if buffer_type == "cbrs":
            self.local_buffer = CBRS(seed=seed,capacity=buffer_size)#CBRS
        elif buffer_type == "rs":
            self.local_buffer = ReservoirSampling(seed=seed,capacity=buffer_size)#reservoir sampling
        elif buffer_type == "wrs":
            self.local_buffer = WRS(seed=seed,capacity=buffer_size)#weighted reservoir sampling
        elif buffer_type == "random":
            self.local_buffer = Random(seed=seed,capacity=buffer_size)#random 0.5
        elif buffer_type == "klrs":
            self.local_buffer = KLRS(seed=seed,capacity=buffer_size,num_classes=K,pi=pi, kl_technique=kl_technique, kl_alpha=kl_alpha )#reservoir sampling
        else:
            self.local_buffer = None
        
        if dataset_name in ['mnist', 'fashion_mnist','emnist']:
            channels = 1
            img_w, img_h = 28, 28
            def forward(x):
                net = hk.Sequential([
                    hk.Flatten(),
                    hk.Linear(250), jax.nn.relu,
                    hk.Linear(250), jax.nn.relu,
                    hk.Linear(K),
                ])
                return net(x)
            self.forward = hk.transform_with_state(forward)
        elif dataset_name in [ 'cifar10', 'cifar100' ]:
            channels = 3
            img_w, img_h = 32, 32
            if nn_architecture == 'resnet18':
                def forward(x, is_training):
                    net = hk.nets.ResNet18(K)
                    return net(x, is_training)
                self.forward = hk.transform_with_state(forward)
            else:
                def forward(x):
                    conv1 = hk.Conv2D( output_channels=64, kernel_shape=(5,5), stride=(1,1), padding="SAME")(x)
                    conv1 = jax.nn.relu(conv1)
                    conv1 = hk.max_pool( conv1, window_shape=(3, 3), strides=(2, 2), padding="SAME" )

                    conv2 = hk.Conv2D( output_channels=64, kernel_shape=(5,5), stride=(1,1), padding="SAME")(conv1)
                    conv2 = jax.nn.relu(conv2)
                    conv2 = hk.max_pool( conv2, window_shape=(3, 3), strides=(2, 2), padding="SAME" )
                    
                    sub_net = hk.Sequential([  
                        hk.Flatten(),
                            hk.Linear(384), jax.nn.relu,
                            hk.Linear(192), jax.nn.relu,
                            hk.Linear( K ),
                        ])
                    return sub_net( conv2 )
                self.forward = hk.transform_with_state(forward)
        
        if nn_architecture == 'resnet18':
            inputs = jnp.zeros((32, 224, 224, channels) )#dummy inputs
            self.theta, self.state = self.forward.init(self.nn_rng, inputs, is_training=True)
            pytorch_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)#pretrained=True)
            pytorch_model.fc = Linear( 512, K )
            self.theta = transfer_params_to_jax( self.theta, pytorch_model )
        else:
            inputs = jnp.zeros((32, img_w, img_h, channels) )#dummy inputs
            self.theta, self.state = self.forward.init(self.nn_rng, inputs)

    def loss_fn( self, params, state, inputs, outputs, is_training=True ):
        if nn_architecture == 'resnet18':
            logits, state = self.forward.apply( params, state, self.nn_rng, inputs, is_training=is_training )
        else:
            logits, state = self.forward.apply( params, state, self.nn_rng, inputs )    
        loss = optax.softmax_cross_entropy(logits=logits, labels=outputs).mean()
        return loss, (logits,state)

    def SetStream(self, stream):
        self.stream = stream

print( "K at agent is", K )

agent = Agent(seed=seed, K=K, stream=agent_stream, dataset_name=dataset_name,
                 buffer_size=local_buffer_size, buffer_type=local_buffer, pi=pi, kl_alpha=kl_alpha, kl_technique=kl_technique )

###------------------Main algorithm------------------###

Test_Accs, Forgettings, Trade_Offs, Pis = [], [], [], []

acc_per_cat=[ set() for i in range(K)  ]
full_acc_per_cat=[ [] for i in range(K)  ]
confusion_matrix = np.zeros( (K, K ) )

t = 0

st = time.time()

while agent.stream.hasNextBatch():

    support_inp, support_labels = agent.stream.GetNextBatch( batch_size )
    
    #mark new categories as seen
    for mark_cat in support_labels:
        cat = mark_cat.tolist()
        #print( type(cat) )#prints int
        if cat not in agent.seen_categories:
            agent.seen_categories[cat] = 1
        #print( agent.seen_categories )

    #ViewBatch( support_inp, support_labels )

    batch_train_inp = deepcopy( support_inp )#copy batch
    batch_train_labels = deepcopy( support_labels )
    support_labels = ToOneHot( support_labels, agent.K )

    query_inp, query_labels = agent.stream.GetTestDataLimitClass( agent.seen_categories )

    if kl_technique == 'dynamic_pi':
        if t == 0:
            agent.local_buffer.SetPi( 1.0 )
        else:
            test_prob = [0]*K
            sum_n = 0
            for category in query_labels:
                sum_n += category.shape[0]
            for category in query_labels:
                test_prob[ category[0].argmax() ] = (category.shape[0]+0.0) / sum_n
            agent.local_buffer.SetPi( agent.local_buffer.CalculatePi( test_prob=test_prob ) )
            #print( 'new pi', agent.local_buffer.pi )

    #train_agent at current batch+replay
    inner_lista = []
    for L in range(inner_steps):

        #gradient of batch
        agent_grad, (_,agent.state) = jax.grad( agent.loss_fn, has_aux=True )( agent.theta, agent.state,
                                        normalize( support_inp, dataset_name ), support_labels, is_training=True )
        if freeze_top and nn_architecture == 'resnet18':
            agent_grad = freeze_top_layers( agent_grad )
        #print( agent_grad )

        has_replay = False
        if local_buffer in ["random","rs","wrs","cbrs","klrs"]:
            #weighted replay
            if replay_method in ['weighted']:
                replay_samples = agent.local_buffer.WeightedSample( replay_size )
            else:
                replay_samples = agent.local_buffer.UniformSample( replay_size )
            if len(replay_samples) > 0:
                has_replay = True
                replay_inp = jnp.array( [ qz[0] for qz in replay_samples ] )#get images
                replay_labels = [ qz[1] for qz in replay_samples ]#get labels
                replay_labels = ToOneHot( replay_labels, agent.K )#convert labels to onehot
            
            if has_replay == True:
                if dynamic_trade_off == True:
                    trade_off = 1.0 / len( agent.seen_categories )
                    
                replay_grad, (_,agent.state) = jax.grad( agent.loss_fn, has_aux=True )( agent.theta, agent.state,
                                            normalize( replay_inp, dataset_name ), replay_labels, is_training=True )
                if freeze_top and nn_architecture == 'resnet18':
                    replay_grad = freeze_top_layers( replay_grad )
                
                agent.theta = jax.tree_map( update_rule1, agent.theta, agent_grad, replay_grad )#update parameters [batch_grad+replay_grad]
            else:
                agent.theta = jax.tree_map( update_rule2, agent.theta, agent_grad )#update parameters [batch_grad]
        else:
            agent.theta = jax.tree_map( update_rule2, agent.theta, agent_grad )#update parameters [batch_grad]
        
    #update buffer - Agent inserts samples to its own local buffer
    if local_buffer in ["random","rs","wrs","cbrs","klrs"]:
        #Reservoir sampling
        #print( batch_train_inp.shape[0] )
        for lb in range( batch_train_inp.shape[0] ):
            if local_buffer in [ 'cbrs', 'rs', 'random', 'klrs' ]:
                agent.local_buffer.Insert( (deepcopy(batch_train_inp[lb]),
                                            deepcopy(batch_train_labels[lb]) ) )
            elif local_buffer == 'wrs':
                datapoint_weight = 1.0
                #print( batch_train_inp[lb].reshape( (1,) + batch_train_inp[lb].shape ).shape )
                reshaped_input = batch_train_inp[lb].reshape( (1,) + batch_train_inp[lb].shape )
                _, (predicted,_) = agent.loss_fn( agent.theta, agent.state,
                                                    normalize(reshaped_input,dataset_name),
                                                    batch_train_labels[lb], is_training=False )
                #print( 'predicted', predicted )
                #input()
                predicted = predicted[0]
                p_x = jnp.exp( predicted - jnp.max(predicted) )
                p_x = p_x / p_x.sum()
                #print( p_x )
                #print( 'softmax predicted', predicted )
                #input()
                for p in p_x:
                    #print( 'p', p )
                    datapoint_weight -= p*p
                #print( datapoint_weight )#, predicted, p_x )
                #print( type(datapoint_weight) )
                agent.local_buffer.Insert( (deepcopy(batch_train_inp[lb]),
                                            deepcopy(batch_train_labels[lb]),
                                            deepcopy(datapoint_weight) ) )

    #Calculate Average accuracy and Forgeting        
    test_acc = 0.0
    forgetting = 0.0

    for i in range( len(query_labels) ):
        
        category = deepcopy( query_labels[i][0].argmax().item() )
        #print( 'category', category)

        #process test into bathches
        evaluate_batch = 32
        correct = 0.0
        for z in range( 0, query_inp[i].shape[0], evaluate_batch ):
            query_inp[z:z+evaluate_batch], 

            _, (cl_test_predicted,_) = agent.loss_fn( agent.theta, agent.state,
                                                normalize(query_inp[i][z:z+evaluate_batch],dataset_name),
                                                query_labels[i][z:z+evaluate_batch], is_training=False )
            correct += jnp.sum( stax.softmax( cl_test_predicted, axis=-1 ).argmax( axis=-1 ) == query_labels[i][z:z+evaluate_batch].argmax( axis=-1 ) )

            if agent.stream.hasNextBatch() == False:
                #calculate confusion matrix
                tl_predicted = stax.softmax( cl_test_predicted, axis=-1 ).argmax( axis=-1 )
                tl_actual = query_labels[i][z:z+evaluate_batch].argmax( axis=-1 )
                for zt in range( len(tl_actual) ):
                    confusion_matrix[ tl_actual[zt] ][ tl_predicted[zt] ] += 1

        correct /= query_inp[i].shape[0]
        
        #does not fit in memory when use_gpu is True
        #_, (cl_test_predicted,_) = agent.loss_fn( agent.theta, agent.state,
        #                                    normalize(query_inp[i],dataset_name), query_labels[i], is_training=False )

        avg_acc = correct

        if len(acc_per_cat[category]) > 0:
            max_diff = max( [ previous_acc-avg_acc for previous_acc in acc_per_cat[category] ] )
            forgetting += max_diff
        
        test_acc += avg_acc
        #print( type(category) )
        acc_per_cat[category].add( avg_acc.item() )
        full_acc_per_cat[category].append( avg_acc.item() )#2d array for statistics
        #print( acc_per_cat )
        #input()
        
    test_acc /= len(query_labels)
    forgetting /= len(query_labels)
    print( 'T', t, 'Test acc', test_acc, 'Forgetting', forgetting, 'Trade-off', trade_off, 'Seen classes', len( agent.seen_categories ) )
    t += 1

    Test_Accs.append( test_acc )
    Forgettings.append( forgetting )
    Trade_Offs.append( trade_off )
    if local_buffer == 'klrs':
        Pis.append( agent.local_buffer.pi )
  
en = time.time()

if dynamic_trade_off:
    trade_off = args.trade_off#the value is important only for loading the files below

print( "Stream Ended")

print( "Total time", en-st )

if kl_technique in [ "dynamic_pi", "target" ]:
    pi = 1.0#the value is important only for loading the files below

#Save Test Acc, Forgetting, Trade-Off
np.save( folder_struct + "%s_Steps%d_TestAccs_arch_%s_LB%s_seed%d_LBsz%d_dto%s_toff%s_repsz%d_repmtd%s_batsz%d_freezetop%s_pi%s_testscen%s_kltech%s_klalpha%s" % (dataset_name, inner_steps, nn_architecture, local_buffer, seed, local_buffer_size, dynamic_trade_off, trade_off, replay_size, replay_method, batch_size, freeze_top, pi, test_scenario, kl_technique, kl_alpha ), Test_Accs )
np.save( folder_struct + "%s_Steps%d_Fgt_arch_%s_LB%s_seed%d_LBsz%d_dto%s_toff%s_repsz%d_repmtd%s_batsz%d_freezetop%s_pi%s_testscen%s_kltech%s_klalpha%s" % (dataset_name, inner_steps, nn_architecture, local_buffer, seed, local_buffer_size, dynamic_trade_off, trade_off, replay_size, replay_method, batch_size, freeze_top, pi, test_scenario, kl_technique, kl_alpha ), Forgettings )
np.save( folder_struct + "%s_Steps%d_TradeOffs_arch_%s_LB%s_seed%d_LBsz%d_dto%s_toff%s_repsz%d_repmtd%s_batsz%d_freezetop%s_pi%s_testscen%s_kltech%s_klalpha%s" % (dataset_name, inner_steps, nn_architecture, local_buffer, seed, local_buffer_size, dynamic_trade_off, trade_off, replay_size, replay_method, batch_size, freeze_top, pi, test_scenario, kl_technique, kl_alpha ), Trade_Offs )
np.save( folder_struct + "%s_Steps%d_Pis_arch_%s_LB%s_seed%d_LBsz%d_dto%s_toff%s_repsz%d_repmtd%s_batsz%d_freezetop%s_pi%s_testscen%s_kltech%s_klalpha%s" % (dataset_name, inner_steps, nn_architecture, local_buffer, seed, local_buffer_size, dynamic_trade_off, trade_off, replay_size, replay_method, batch_size, freeze_top, pi, test_scenario, kl_technique, kl_alpha ), Pis )
#save stats
stats = []
stats.append( ( 'stream_labels', [ j.item() for j in agent.stream.train_stream_labels] ) )
stats.append( ( 'total_time', en-st ) )
stats.append( ('full_acc_per_cat', full_acc_per_cat) )
stats.append( ('confusion_matrix', confusion_matrix) )
if local_buffer != "nobuff":
    stats.append( ( 'buffer_labels', [ j[1] for j in agent.local_buffer.q ] ) )
    stats.append( ( 'buffer_counts', agent.local_buffer.m ) )
    stats.append( ( 'stream_counts', agent.local_buffer.n ) )
buffer_file = open( folder_struct + "%s_Steps%d_Stats_arch_%s_LB%s_seed%d_LBsz%d_dto%s_toff%s_repsz%d_repmtd%s_batsz%d_freezetop%s_pi%s_testscen%s_kltec%s_klalpha%s.txt" % (dataset_name, inner_steps, nn_architecture, local_buffer, seed, local_buffer_size, dynamic_trade_off, trade_off, replay_size, replay_method, batch_size, freeze_top, pi, test_scenario, kl_technique, kl_alpha ), "wb")
pickle.dump( stats, buffer_file)
buffer_file.close()