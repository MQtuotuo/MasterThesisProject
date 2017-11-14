import numpy as np
import argparse
import traceback
import os
import keras
np.random.seed(1337)  # for reproducibility
keras.backend.set_image_dim_ordering('tf')

import util
import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Path to data dir')
    # TODO: check is correct
    parser.add_argument('--model', type=str, required=True, help='Base model architecture',
                        choices=[config.MODEL_RESNET50, config.MODEL_RESNET152, config.MODEL_INCEPTION_V3,
                                 config.MODEL_VGG16, config.MODEL_BONET, config.MODEL_INCEPTION_RESNET_V2])
    parser.add_argument('--nb_epoch', type=int, default=200)
    parser.add_argument('--freeze_layers_number', type=int, help='will freeze the first N layers and unfreeze the rest')
    parser.add_argument('--gender', type=str, required=True, help='The gender of patient, M or F or ALL')
    return parser.parse_args()


if __name__ == '__main__':
    try:
        args = parse_args()

        if args.data_dir:
            config.data_dir = args.data_dir
            config.set_paths()
        if args.model:
            config.model = args.model

        print(config.get_classes_path())
        print(config.get_model_path())
        util.lock()
        util.override_keras_directory_iterator_next()
        #util.set_classes_from_train_dir()
        util.set_samples_info()
        if not os.path.exists(config.trained_dir):
            os.mkdir(config.trained_dir)

        #class_weight = util.get_class_weight(config.train_male_dir)
        #print(class_weight)

        # TODO: create class instance without dynamic module import
        model = util.get_model_class_instance(
            #class_weight=class_weight,
            nb_epoch=args.nb_epoch,
            freeze_layers_number=args.freeze_layers_number,
            gender=args.gender)

        print(model)
        model.train()
        print('Training is finished!')
    except (KeyboardInterrupt, SystemExit):
        util.unlock()
    except Exception as e:
        print(e)
        print(traceback.format_exc())
    util.unlock()
