# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 06:49:08 2019

@author: Shrishti D Hore
"""

#importing required libraries and packages
import mxnet as mx
from mxnet import gluon
import utils
import glob
import scipy
from scipy import signal
from matplotlib import pyplot as plt
from keras.models import model_from_json
from matplotlib import colors
import glob
import numpy as np
import os
from PIL import Image

#We are using Convolutional Autoencoder(CAE) for good reason that they can be trained in normal parts and wont require annotated data.
#Once trained we can give a featured representation for a part and compare autoencoder output and input
#Concept is the larger the difference the more likely the anomaly
#Two parts of the autoencoder : encoder and decoder
#Here encoder will encode the input data using a reduced representation and the decoder will attempt to re-construct the original input data from reduced representation
#In our CAE model encoder will consist of 2 convolutional and 2 Max-Pooling layers and decoder will consists of two Unsampling and Two Deconvolutions

# convolutional autoencoder
class ConvolutionalAutoencoder(gluon.nn.HybridBlock):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        with self.name_scope():
            self.encoder = gluon.nn.HybridSequential(prefix="")
            with self.encoder.name_scope():
                self.encoder.add(gluon.nn.Conv2D(32, 5, padding=0, activation='relu'))
                self.encoder.add(gluon.nn.MaxPool2D(2))
                self.encoder.add(gluon.nn.Conv2D(32, 5, padding=0, activation='relu'))
                self.encoder.add(gluon.nn.MaxPool2D(2))
                self.encoder.add(gluon.nn.Dense(2000))
            self.decoder = gluon.nn.HybridSequential(prefix="")
            with self.decoder.name_scope():
                self.decoder.add(gluon.nn.Dense(32*22*22, activation='relu'))
                self.decoder.add(gluon.nn.HybridLambda(lambda F, x: F.UpSampling(x, scale=2, sample_type='nearest')))
                self.decoder.add(gluon.nn.Conv2DTranspose(32, 5, activation='relu'))
                self.decoder.add(gluon.nn.HybridLambda(lambda F, x: F.UpSampling(x, scale=2, sample_type='nearest')))
                self.decoder.add(gluon.nn.Conv2DTranspose(1, kernel_size=5, activation='sigmoid'))


    def hybrid_forward(self, F, x):
        x = self.encoder(x)
        x = self.decoder[0](x)
        # need to reshape ouput feature vector from Dense(32*22*22), before it is upsampled
        x = x.reshape((-1,32,22,22))
        x = self.decoder[1:](x)

        return x

# Train the autoencoder
def train(batch_size, ctx, num_epochs, path, lr=1e-4, wd=1e-5, params_file="autoencoder_ucsd_convae.params"):

  # Dataloader for training dataset
  dataloader = utils.create_dataset(path, batch_size, shuffle=True)

  # Get model
  model = ConvolutionalAutoencoder()
  model.hybridize()

  # Initialiize
  model.collect_params().initialize(mx.init.Xavier('gaussian'), ctx=ctx)
  
  # Loss
  l2loss = gluon.loss.L2Loss()
  optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})

  # Start training loop
  for epoch in range(num_epochs):
    for image in dataloader:
        image = image.as_in_context(ctx)

        with mx.autograd.record():
            reconstructed = model(image)
            loss = l2loss(reconstructed, image)

            loss.backward()
            optimizer.step(batch_size)
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, mx.nd.mean(loss).asscalar()))

  # Save parameters
  model.save_parameters(params_file)
  return model, params_file

'''
  for epoch in range(num_epochs):
    
      for image_batch in dataloader:
        
          image = image_batch.as_in_context(ctx)

          with mx.autograd.record():
              output = model(image_batch)
              loss = loss_function(output, image_batch)

          loss.backward()
          optimizer.step(image.shape[0])
          print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, mx.nd.mean(loss).asscalar()))

          model.save_parameters("autoencoder_ucsd.params")


          files = sorted(glob.glob('HEU_AI/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test024/*'))

          a = np.zeros((len(files),1,100,100))

for idx,filename in enumerate(files):
    im = Image.open(filename)
    im = im.resize((100,100))
    a[idx,0,:,:] = np.array(im, dtype=np.float32)/255.0

dataset = gluon.data.ArrayDataset(mx.nd.array(a, dtype=np.float32))
dataloader = gluon.data.DataLoader(dataset, batch_size=1)

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
