from os.path import join as join_path
import os

abspath = os.path.dirname(os.path.abspath(__file__))

lock_file = os.path.join(abspath, 'lock')

data_male_dir = join_path(abspath, 'RSNA/sorted/male')
data_female_dir = join_path(abspath, 'RSNA/sorted/female')

trained_dir = join_path(abspath, 'trained')

train_male_dir, validation_male_dir = None, None
train_female_dir, validation_female_dir = None, None

MODEL_VGG16 = 'vgg16'
MODEL_INCEPTION_V3 = 'inception_v3'
MODEL_RESNET50 = 'resnet50'
MODEL_RESNET152 = 'resnet152'
MODEL_BONET = 'boNet'
MODEL_INCEPTION_RESNET_V2 = 'inception_resnet'

model = MODEL_RESNET50

bf_train_path = join_path(trained_dir, 'bottleneck_features_train.npy')
bf_valid_path = join_path(trained_dir, 'bottleneck_features_validation.npy')
top_model_weights_path = join_path(trained_dir, 'top-model-{}-weights.h5')
fine_tuned_weights_path = join_path(trained_dir, 'fine-tuned-{}-weights.h5')
model_path = join_path(trained_dir, 'model-{}.h5')
classes_path = join_path(trained_dir, 'classes-{}')

activations_path = join_path(trained_dir, 'activations.csv')
novelty_detection_model_path = join_path(trained_dir, 'novelty_detection-model-{}')

plots_dir = join_path(abspath, 'plots')

# server settings
server_address = ('0.0.0.0', 4224)
buffer_size = 4096

classes = []

nb_train_samples = 0
nb_validation_samples = 0


def set_paths():
    global train_male_dir, validation_male_dir, train_female_dir, validation_female_dir
    train_male_dir = join_path(data_male_dir, 'train/')
    validation_male_dir = join_path(data_male_dir, 'valid/')
    train_female_dir = join_path(data_female_dir, 'train/')
    validation_female_dir = join_path(data_female_dir, 'valid/')


set_paths()


def get_top_model_weights_path():
    return top_model_weights_path.format(model)


def get_fine_tuned_weights_path(checkpoint=False):
    return fine_tuned_weights_path.format(model + '-checkpoint' if checkpoint else model)


def get_novelty_detection_model_path():
    return novelty_detection_model_path.format(model)


def get_model_path():
    return model_path.format(model)


def get_classes_path():
    return classes_path.format(model)