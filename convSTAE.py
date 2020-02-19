# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 07:13:06 2019

@author: Shrishti D Hore
"""

import mxnet as mx
from mxnet import gluon
import numpy as np 
from mxnet.gluon import nn
import utils
import scipy
from scipy import signal
import mxnet as mx
from matplotlib import pyplot as plt
from keras.models import model_from_json
from matplotlib import colors
from mxnet import gluon
import glob
import numpy as np
import os
from PIL import Image

# convolutional autoencoder with convolutional LSTMs
class convLSTMAE(gluon.nn.HybridBlock):
    def __init__(self, **kwargs):
        super(convLSTMAE, self).__init__(**kwargs)
        with self.name_scope():

          self.encoder = gluon.nn.HybridSequential()
          self.encoder.add(gluon.nn.Conv2D(128, kernel_size=11, strides=4, activation='relu'))
          self.encoder.add(gluon.nn.Conv2D(64, kernel_size=5, strides=2, activation='relu'))

          self.temporal_encoder = gluon.rnn.HybridSequentialRNNCell()
          self.temporal_encoder.add(gluon.contrib.rnn.Conv2DLSTMCell((64,26,26), 64, 3, 3, i2h_pad=1))
          self.temporal_encoder.add(gluon.contrib.rnn.Conv2DLSTMCell((64,26,26), 32, 3, 3, i2h_pad=1))
          self.temporal_encoder.add(gluon.contrib.rnn.Conv2DLSTMCell((32,26,26), 64, 3, 3, i2h_pad=1))

          self.decoder =  gluon.nn.HybridSequential()
          self.decoder.add(gluon.nn.Conv2DTranspose(channels=128, kernel_size=5, strides=2, activation='relu'))
          self.decoder.add(gluon.nn.Conv2DTranspose(channels=10, kernel_size=11, strides=4, activation='sigmoid'))

    def hybrid_forward(self, F, x, states=None, **kwargs):
        x = self.encoder(x)
        x, states = self.temporal_encoder(x, states)
        x = self.decoder(x)

        return x, states

# Train the autoencoder
def train(batch_size, ctx, num_epochs, path, lr=1e-4, wd=1e-5, params_file="autoencoder_ucsd_convLSTMAE.params"):

  # Dataloader for training dataset
  dataloader = utils.create_dataset_stacked_images(path, batch_size, shuffle=True, augment=True)

  # Get model
  model = convLSTMAE()
  model.hybridize()
  model.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
  states = model.temporal_encoder.begin_state(batch_size=batch_size, ctx=ctx)

  # Loss
  l2loss = gluon.loss.L2Loss()
  optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd, 'epsilon':1e-6})

  # start the training loop
  for epoch in range(num_epochs):
    for image in dataloader:

        image  = image.as_in_context(ctx)
        states = model.temporal_encoder.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=ctx)
        
        with mx.autograd.record():
            reconstructed, states = model(image, states)
            loss = l2loss(reconstructed, image)

            loss.backward()
            optimizer.step(batch_size)
        
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, mx.nd.mean(loss).asscalar()))

  #save parameters
  model.save_parameters(params_file)

  return model, params_file

'''
def plot(img, output, diff, H, threshold, counter):
    
    fig, (ax0, ax1, ax2,ax3) = plt.subplots(ncols=4, figsize=(10, 5))
    ax0.set_axis_off()
    ax1.set_axis_off()
    ax2.set_axis_off()
    
    ax0.set_title('input image')
    ax1.set_title('reconstructed image')
    ax2.set_title('diff ')
    ax3.set_title('anomalies')
    
    ax0.imshow(img, cmap=plt.cm.gray, interpolation='nearest') 
    ax1.imshow(output, cmap=plt.cm.gray, interpolation='nearest')   
    ax2.imshow(diff, cmap=plt.cm.viridis, vmin=0, vmax=255, interpolation='nearest')  
    ax3.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    
    x,y = np.where(H > threshold)
    ax3.scatter(y,x,color='red',s=0.1) 

    plt.axis('off')
    
    fig.savefig('images/' + str(counter) + '.png')
    
threshold = 4*255
counter = 0
try:
    os.mkdir("images")
except:
    pass

for image in dataloader:
    
    counter = counter + 1
    img = image.as_in_context(mx.cpu())

    output = model(img)
    output = output.transpose((0,2,3,1))
    image = image.transpose((0,2,3,1))
    output = output.asnumpy()*255
    img = image.asnumpy()*255
    diff = np.abs(output-img)
    
    tmp = diff[0,:,:,0]
    H = signal.convolve2d(tmp, np.ones((4,4)), mode='same')
 
    plot(img[0,:,:,0], output[0,:,:,0], diff[0,:,:,0], H, threshold, counter)
    break 
'''