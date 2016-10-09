allfiles = os.listdir(data_path)
allfiles.sort()
p = re.compile(filename_regex)
eventlogfiles = list(filter(lambda x: p.match(x) is not None, allfiles))
# coding: utf-8

# In[29]:

UNIT_TESTS=True


# In[30]:



# Theano isn't working with GPU. http://stackoverflow.com/q/38125310/266387
#os.environ['KERAS_BACKEND'] = 'theano'


# In[31]:

import pandas as pd
import numpy as np
from random import random
import math
import re
import os
import io
import calendar
import hashlib


from IPython.display import display, HTML

np.random.seed()


# In[32]:




# In[36]:


# In[96]:

import unittest



# In[97]:

from IPython.display import display, HTML

if True:  #UNIT_TESTS:    
    bs = BatchSample('/home/greg/Projects/Software/TimeTillComplete.com/data/population_v2.1/2005-11-11.log', 0, 1)
    #HTML(bs.dfX.to_html())
    print(bs.dfy)
else:
    print('pass')
    


# In[98]:

if UNIT_TESTS:
    



# In[38]:


# <h3>Hand training</h3>

# In[39]:

SAMPLE_FILES_PATH='/home/greg/Projects/Software/TimeTillComplete.com/data/population_v2.1/'
NP_DATA_FILE='/home/greg/Projects/Software/TimeTillComplete.com/data/population_v2.1/WIP_08.npz'
MODEL_SAVE_FILE='/home/greg/Projects/Software/TimeTillComplete.com/data/WIP_08.model'


# In[40]:

modelData = TTCModelData()


# In[41]:

modelData.load_raw_sample_files(SAMPLE_FILES_PATH)

print("training_files   :%d" % (len(modelData.training_files)))
print("validation_files :%d" % (len(modelData.validation_files)))
print("testing_files    :%d" % (len(modelData.testing_files)))


# In[42]:

modelData.save_np_data_files(NP_DATA_FILE)


# In[43]:

#modelData.load_np_data_files(NP_DATA_FILE)


# In[44]:



try:
    import dill as pickle
    print('Went with dill')
except ImportError:
    import pickle


# In[45]:

import matplotlib
#matplotlib.use('Agg')


# In[46]:





# In[50]:

#import importlib
#importlib.reload(matplotlib)
#importlib.reload(matplotlib.pyplot)


# In[51]:

nb_samples     = modelData.X_train.shape[0]
timesteps      = modelData.X_train.shape[1]
input_dim      = modelData.X_train.shape[2]

in_neurons     = 555
hidden_layers  = 3
hidden_neurons = 555
out_neurons    = in_neurons
rnn_activation = 'relu'
dense_activation = 'linear'

batch_size     = timesteps
epochs = 27


# In[52]:



# In[54]:



# In[ ]:

for idx in range(1):
    print('Starting epoch loop %d' % (idx))
    training_history = model.fit(
              modelData.X_train,
              modelData.y_train,
              nb_epoch        = epochs,
              batch_size      = batch_size,
              shuffle         = False,
              validation_data = (modelData.X_validation, modelData.y_validation),
              verbose         = 1,
            )
    #bokehplot_losses(training_history.history)


# In[26]:

# No wide Dense layer
#batch_size      : 385
#nb_samples      : 59015
#timesteps       : 55
#input_dim       : 52
#in_neurons      : 333
#hidden_layers   : 2
#hidden_neurons  : 333
#out_neurons     : 333
#rnn_activation  : tanh
#dense_activation: linear
#333,2,333 scores 4056.2156862696061, 0.0

# No wide Dense layer
#batch_size      : 385
#nb_samples      : 59015
#timesteps       : 55
#input_dim       : 52
#in_neurons      : 555
#hidden_layers   : 2
#hidden_neurons  : 555
#out_neurons     : 555
#rnn_activation  : tanh
#dense_activation: linear
#scores: [4050.4207381049437, 0.0]
#Model Accuracy: 0.00%


# Extra Dense(in_neurons)
#batch_size      : 385
#nb_samples      : 59015
#timesteps       : 55
#input_dim       : 52
#in_neurons      : 555
#hidden_layers   : 2
#hidden_neurons  : 555
#out_neurons     : 555
#rnn_activation  : tanh
#dense_activation: linear
#scores: [1325.3205331643626, 0.0]
#Model Accuracy: 0.00%

## Single layer LSTM
#batch_size      : 55
#nb_samples      : 59015
#timesteps       : 55
#input_dim       : 52
#in_neurons      : 222
#hidden_layers   : 0
#hidden_neurons  : 222
#out_neurons     : 222
#rnn_activation  : tanh
#dense_activation: linear
#scores: [4615.3364449280016, 0.0]
#Model Accuracy: 0.00%
    
print('batch_size      : %s' % batch_size     )
print('nb_samples      : %s' % nb_samples     )
#print('channels        : %s' % channels       )
print('timesteps       : %s' % timesteps      )
print('input_dim       : %s' % input_dim      )
print('in_neurons      : %s' % in_neurons     )
print('hidden_layers   : %s' % hidden_layers  )
print('hidden_neurons  : %s' % hidden_neurons )
print('out_neurons     : %s' % out_neurons    )
print('rnn_activation  : %s' % rnn_activation )
print('dense_activation: %s' % dense_activation)

scores = model.evaluate(modelData.X_test, modelData.y_test, verbose=0)
print('scores: %s' % str(scores))
print("Model Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:

#matplotlib.use('Agg')
bokehplot_losses(training_history.history)


# In[ ]:

#1537.5439
            )
    bokehplot_losses(training_history.history)


# In[ ]:

model.save(MODEL_SAVE_FILE)


# In[ ]:

#model = keras.model.load(MODEL_SAVE_FILE)
#modelData = TTCModelData.load_np_data_files(NP_DATA_FILE)


# <h2> Prediction Plots from Best Model</h2>

# In[ ]:



# In[ ]:



# In[ ]:




# In[ ]:

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


# In[ ]:

plotDebugBatchSample(modelData, debugBatchSample)


# In[ ]:


plotPredictions(predictions)


# In[ ]:




# In[ ]:

assert type(np.ones((2,2))) == np.ndarray


# In[ ]:

