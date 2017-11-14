from keras.layers import (Flatten, Dense, Dropout, Conv2D, MaxPooling2D, Activation)
from keras.layers import Input
from keras import applications
from keras.layers import Input
from keras.optimizers import SGD
from keras.models import Sequential, Model
import config
import numpy as np
from models.base_model import BaseModel
from models.spatial_transformer import SpatialTransformer
from models.spatial_transformer_network import STN


class boNet(BaseModel):

    def __init__(self, *args, **kwargs):
        super(boNet, self).__init__(*args, **kwargs)
        if not self.freeze_layers_number:
            # we train all the layers
            self.freeze_layers_number = 0


    def _create(self):

        b = np.zeros((2, 3), dtype='float32')
        b[0, 0] = 1
        b[1, 1] = 1
        W = np.zeros((50, 6), dtype='float32')
        weights = [W, b.flatten()]
        data = Input(shape=(224, 224, 3))

        base_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=self.get_input_tensor())
        for i in range(4):
            base_model.layers.pop()

        new_model = Sequential()  # new model as localisation network
        for layer in base_model.layers:
            new_model.add(layer)
        for layer in new_model.layers:
            layer.trainable = False

        # base_model.summary()
        # new_model.summary()
        #locnet
        x = new_model.output
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(50, activation='tanh')(x)
        x = Dense(6, weights=weights)(x)

        locnet = Model(input=new_model.input, output=x)
        locnet.summary()

        # new model
        ST = SpatialTransformer(localization_net=locnet, output_size=(14, 14), input_shape=(224, 224, 3))
        self.model = Sequential()
        self.model.add(ST)
        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2048, activation='elu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.summary()



def inst_class(*args, **kwargs):
    return boNet(*args, **kwargs)

