# Anomaly Detection In Real Time Surveillance Videos using AutoEncoders

This projects detect Anomalous Behavior through live CCTV camera feed to alert the police or local authority for faster response time.

### Code and Resources Used

**Language:** Python 3.8

**Libraries and Modules:** pandas, numpy,  Scipy, argparse, csv, re, cv2, os, glob, io, tensorflow, PIL, shutil, urllib, files(google colab), 

**Dataset:** [Avenue Dataset for Abnormal Detection](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)

**Keywords:** Anomaly Detection, Spatio Temporal AutoEncoder, Computer Vision

**Step 1: Data Pre-Processing**

1. Download the videos ie; 16 training videos and 12 testing videos and divide it by frames.
2. Images with random objects in the backgorund.
3. Various background conditions such as dark, light, indoor, oudoor, etc.
4. Save all the images in a folder called images and all images should be in .jpg format.
5. Use Argprase parser to add argument to the file names.
6. Divide each and every video into frames and save the frames in a directory separated by the type of anomaly or situation as well as resize the images to scale.
7. Reshape and normalize the images.
8. Clip negative values and remove buffer directory.

**Step 2: Loading the Keras Models**

1. Import the three models given below:
- Convolutional 3D
- Convolutional LSTM 2D
- Convolutional 3D Transpose
2. Using Sequential define filters, padding and activation of these models. I am choosing Relu.
3. Let the optimizer be Adam and metric loss be Categorical Crossentropy.

**AutoEncoder**

- Autoencoder is an unsupervised artificial neural network that learns how to efficiently compress and encode data then learns how to reconstruct the data back from the reduced encoded representation to a representation that is as close to the original input as possible.
- Autoencoder, by design, reduces data dimensions by learning how to ignore the noise in the data.

![autoencoder](https://github.com/ShrishtiHore/Anomaly-Detection-in-CCTV-Surveillance-Videos/blob/master/visualizations/auto.png)

**Autoencoder Components:**

Autoencoders consists of 4 main parts:

1- Encoder: In which the model learns how to reduce the input dimensions and compress the input data into an encoded representation.

2- Bottleneck: which is the layer that contains the compressed representation of the input data. This is the lowest possible dimensions of the input data.

3- Decoder: In which the model learns how to reconstruct the data from the encoded representation to be as close to the original input as possible.

4- Reconstruction Loss: This is the method that measures measure how well the decoder is performing and how close the output is to the original input.

- The training then involves using back propagation in order to minimize the network’s reconstruction loss.
- Architecture : The network architecture for autoencoders can vary between a simple FeedForward network, LSTM network or Convolutional Neural Network depending on the use case.

**Step 3: Training the Model**
1. train.py which runs the training process
2. pipeline_config_path=Path/to/config/file/model.config
3. model_dir= Path/to/training/
4. If the kernel dies, the training will resume from the last checkpoint. Unless you didn’t save the training/ directory somewhere, ex: GDrive.
5. If you are changing the below paths, make sure there is no space between the equal sign = and the path.
6. And use early Callbacks to stop the training if it goes out of hand.

**Step 4: Export the Trained Model**
1. the model will save a checkpoint every 600 seconds while training up to 5 checkpoints. Then, as new files are created, older files are deleted.
2. A file called model.h5 is created which will be used while testing later.
3. Epochs were used as arg.epoch and batch size for training was 32
4. Another file called training.npy would be created it contains the array form of all the coordinates required while testing. SO here no frozen inference graph or pdtxt file is created.

**Step 5: Testing the Detector**
1. Load the model.h5 file and training.npy file.
2. Test the Videos as: 
- Anomalous Bunch of ___ Number
- Whether it is normal or abnormal

**Results**
![result](https://github.com/ShrishtiHore/Anomaly-Detection-in-CCTV-Surveillance-Videos/blob/master/Results/anomaly.PNG)

**References**

1. https://towardsdatascience.com/auto-encoder-what-is-it-and-what-is-it-used-for-part-1-3e5c6f017726#:~:text=Autoencoder%20is%20an%20unsupervised%20artificial,the%20original%20input%20as%20possible.
2. https://github.com/aninair1905/Abnormal-Event-Detection
3. https://arxiv.org/pdf/1701.01546.pdf
4. https://blog.keras.io/building-autoencoders-in-keras.html
5. https://towardsdatascience.com/5-ways-to-detect-outliers-that-every-data-scientist-should-know-python-code-70a54335a623





