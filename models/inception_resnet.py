from inception_resnet_v2 import create_inception_resnet_v2
from keras.layers import (Flatten, Dense, Dropout)
from keras.models import Model

import config
from .base_model import BaseModel

class InceptionNetV2(BaseModel):
    noveltyDetectionLayerName = 'fc1'
    noveltyDetectionLayerSize = 2048

    def __init__(self, *args, **kwargs):
        super(InceptionNetV2, self).__init__(*args, **kwargs)
        if not self.freeze_layers_number:
            # we chose to train the top 2 identity blocks and 1 convolution block
            self.freeze_layers_number = 80

    def _create(self):
        base_model = inception_resnet_v2(include_top=False, input_tensor=self.get_input_tensor())
        self.make_net_layers_non_trainable(base_model)
        x = base_model.output
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        x = Dropout(0.5)(x)
        # for classification: predictions = Dense(len(config.classes), activation='softmax', name='predictions')(x)
        predictions = Dense(1, name='predictions')(x)
        self.model = Model(input=base_model.input, output=predictions)


def inst_class(*args, **kwargs):
    return InceptionNetV2(*args, **kwargs)
