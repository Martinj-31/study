'''
@author: Martin
KIST
'''
from brian2 import *

import sys, os
os.chdir("/Users/mingyucheon/Desktop")
sys.path.append(os.getcwd())

import matplotlib
import numpy as np
import pickle
import datetime
import struct

np.set_printoptions(threshold=sys.maxsize, linewidth=200)

####################################################################
# %% MNIST data load
def getData(filename, training=True):

    MNIST_data_path = "/Users/mingyucheon/Desktop/dataset/"

    if training:
        images = open(MNIST_data_path + 'train-images.idx3-ubyte', 'rb')
        labels = open(MNIST_data_path + 'train-labels.idx1-ubyte', 'rb')
        print("Training data set is decoded.")
    else:
        images = open(MNIST_data_path + 't10k-images.idx3-ubyte', 'rb')
        labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte', 'rb')
        print("Test data set is decoded.")

    # Get metadata for images
    images.read(4) # skip the magic number
    number_of_images = struct.unpack('>I', images.read(4))[0]
    rows = struct.unpack('>I', images.read(4))[0]
    cols = struct.unpack('>I', images.read(4))[0]

    # Get metadata for labels
    labels.read(4) # skip the magic number
    N = struct.unpack('>I', labels.read(4))[0]

    if number_of_images != N:
        raise Exception('The number of labels did not match the number of images')

    # Get the data
    x = np.zeros((N, rows, cols), dtype=np.uint8)
    y = np.zeros((N, 1), dtype=np.uint8)

    for i in range(N):
        if 0 == i % 1000:
            print("i : ", i+1000)
        x[i] = [[struct.unpack('>B', images.read(1))[0] for unused_col in range(cols)] for unused_row in range(rows)]
        y[i] = struct.unpack('>B', labels.read(1))[0]

    data = {'image': x, 'label': y, 'rows': rows, 'cols': cols}
    pickle.dump(data, open(MNIST_data_path + '%s.pickle' % filename, 'wb'))

    # print("Get labeled data takes", round(time.time()-start, 5), "seconds.")
    return data

def setData():
    testSet = getData('MNISTtestset', training=False)
    trainingSet = getData('MNISTtrainingset', training=True)

    return testSet, trainingSet

testSet, trainingSet = setData()
x_test = testSet['image']
y_test = testSet['label']
x_training = trainingSet['image']
y_training = trainingSet['label']


####################################################################
# %% Variables setup
nInput = 784
nHidden = 100
nOutput = 10
nImage = 2

tau = 10*ms
w = 0.2
wmax = 5
Apre = 2
Apost = Apre
duration = 1*second

eqs_neu = '''
dv/dt = -v/tau : 1
tau : second
'''
stdp = '''
w : 1
dapre/dt = -apre/tau : 1 (event-driven)
dapost/dt = -apost/tau : 1 (event-driven)
'''
on_pre = '''
v_post += w
apre = Apre
w = clip(w+apost,0,wmax)
'''
on_post = '''
apost = Apost
w = clip(w+apre,0,wmax)
'''


####################################################################
# %% Network setup
for i in range(10):
    print("#", i+1)
    print(x_training[i])
    P = PoissonGroup(nInput, rates=x_training[i].flatten()*Hz)

    hidden = NeuronGroup(N=nHidden, model=eqs_neu, threshold='v>128', reset='v=0', refractory=5*ms, method='exact')
    STM_hidden = StateMonitor(hidden, 'v', record=True)
    SPM_hidden = SpikeMonitor(hidden)
    output = NeuronGroup(N=nOutput, model=eqs_neu, threshold='v>95', reset='v=0', refractory=5*ms, method='exact')
    STM_output = StateMonitor(output, 'v', record=True)
    SPM_output = SpikeMonitor(output)

    syn_1 = Synapses(source = P, target = hidden, model = stdp, on_pre = on_pre, on_post = on_post)
    syn_1.connect()

    syn_2 = Synapses(source = hidden, target = output, model = stdp, on_pre = on_pre, on_post = on_post)
    syn_2.connect()


    N = Network(P, hidden, output, STM_hidden, SPM_hidden, STM_output, STM_output, syn_1, syn_2)
    N.run(500*ms)

print(SPM_output[0])
####################################################################
# %% Result

plot(M.t/ms, M.v[0])
xlabel('Time (ms')
ylabel('v')
show()
