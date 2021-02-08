from keras.layers import Conv3D,ConvLSTM2D,Conv3DTranspose
from keras.models import Sequential


def loadModel():

    """
    Sequential Model for the Spatio Temporal Autoencoder (STModel)
    :return:
    """

    STModel = Sequential()

    STModel.add(Conv3D(filters=128, kernel_size=(11, 11, 1), strides=(4, 4, 1), padding='valid', input_shape=(227, 227, 10, 1), activation='relu'))
    STModel.add(Conv3D(filters=64, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding='valid', activation='relu'))

    STModel.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', dropout=0.4, recurrent_dropout=0.3, return_sequences=True))
    STModel.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', dropout=0.3, return_sequences=True))
    STModel.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, return_sequences=True, padding='same', dropout=0.5))

    STModel.add(Conv3DTranspose(filters=128,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='relu'))
    STModel.add(Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',activation='relu'))


    STModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return STModel
