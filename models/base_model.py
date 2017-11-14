from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2gray
from skimage import data
from keras.layers import (Flatten, Dense, Dropout)
from keras.layers import Input
from keras.optimizers import SGD
from keras.models import Sequential, Model
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from skimage import transform as tf
from scipy import misc
from keras.optimizers import Adam
import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard, ReduceLROnPlateau
import config
import pandas
import os
from util import save_history
import util
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import cv2
from pseudoRGB import pseudoRGB
from resizeToFit import *
import keras.backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from data import load_train_data, load_test_data
from preprocessing import preprocessing, postprocessing, clahe_augment, contrast_augment



class BaseModel(object):
    def __init__(self,
                 class_weight=None,
                 nb_epoch=1000,
                 freeze_layers_number=None,
                 gender=None):
        self.model = None
        self.class_weight = class_weight
        self.nb_epoch = nb_epoch
        self.fine_tuning_patience = 20
        self.batch_size = 32
        self.freeze_layers_number = freeze_layers_number
        self.img_size = (224, 224)
        self.imageShape = (224, 224, 3)
        self.gender = gender

    def _create(self):
        raise NotImplementedError('subclasses must override _create()')

    def _fine_tuning(self):
        seed = 7
        self.freeze_top_layers()

        # for regression task
        self.model.compile(loss='mean_squared_error', metrics=['mae'], optimizer=Adam(lr=1.0e-5))
        #self.model.compile(loss=self.huber_loss, metrics=['mae'], optimizer='adam')
        #self.model.load_weights(config.get_fine_tuned_weights_path())

        #X_train, y_train, X_test, y_test = self.getData()
        '''
        X, Y= self.getData()
   
        np.random.seed(seed)
       

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        CVscores = []
        for train, test in kfold.split(X, Y):
           
            callbacks = self.get_callbacks(config.get_fine_tuned_weights_path())
            self.model.compile(loss='mean_squared_error', metrics=['mae'], optimizer=Adam(lr=1.0e-5))
            # Fit the model
            # fits the model on batches with real-time data augmentation:
            train_data = self.get_train_datagen(X[train], Y[train])
            valid_data = self.get_validation_datagen(X[test], Y[test])
            self.model.fit_generator(
                train_data,
                steps_per_epoch=len(X[train]) / float(self.batch_size), 
                validation_data=valid_data,
                validation_steps=len(X[test]) / float(self.batch_size),
                epochs=self.nb_epoch, 
                callbacks=callbacks)

            #self.model.fit(X[train], Y[train], epochs=self.nb_epoch, batch_size=self.batch_size, validation_split=0.2, callbacks=callbacks)
            # evaluate the model
            scores = self.model.evaluate(X[test], Y[test], verbose=0)
            CVscores.append(scores)
            print("validation MSE: ", scores)
        CVmean = np.mean(CVscores)
        CVstd = np.std(CVscores)
        CVpct = 100 * CVstd / CVmean
        print("For " + str(n_fold) + "-fold CV, avg MSE: " + format(CVmean, '.4f') +", std dev of MSE: " + format(CVstd, '.4f') +" (" + format(CVpct,'.2f') + "%)")

        '''

        X, Y= self.getData()
        X, Y = shuffle(X, Y, random_state=seed)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)  

        train_data = self.get_train_datagen(X_train, y_train)
        valid_data = self.get_validation_datagen(X_test, y_test)
        callbacks = self.get_callbacks(config.get_fine_tuned_weights_path())

        history = self.model.fit_generator(
                train_data,
                steps_per_epoch=len(X_train) / float(self.batch_size), 
                validation_data=valid_data,
                validation_steps=len(X_test) / float(self.batch_size),
                epochs=self.nb_epoch,
                callbacks=callbacks)
        
       
        self.model.save(config.get_model_path())

        save_history(history, 'history_finetuning.txt')

        print(history.history.keys())
        scores = self.model.evaluate(X_test, y_test, verbose=1)
        print('scores on test data', scores)

        scores_t = self.model.evaluate(X_train, y_train, verbose=1)
        print('scores on train data', scores_t)

        # Print MAE 
        print ("Model MAE (test data): ", mean_absolute_error(self.model.predict(X_test), y_test))
        #print ("Model MAE (valid data): ", mean_absolute_error(self.model.predict(X_valid), y_valid))    
        print ("Model MAE (train data): ", mean_absolute_error(self.model.predict(X_train), y_train))
    
        # Print MSE 
        print ("Model MSE (test data): ", mean_squared_error(self.model.predict(X_test), y_test))
        #print ("Model MSE (valid data): ", mean_squared_error(self.model.predict(X_valid), y_valid))
        print ("Model MSE (train data): ", mean_squared_error(self.model.predict(X_train), y_train))

        # Print R^2 
        print ("Model R^2 (test data): ", r2_score(self.model.predict(X_test), y_test))
        #print ("Model R^2 (valid data): ", r2_score(self.model.predict(X_valid), y_valid))
        print ("Model R^2 (train data): ", r2_score(self.model.predict(X_train), y_train))
        


    def train(self):
        print("Creating model...")
        self._create()
        print("Model is created")
        print("Fine tuning...")
        self._fine_tuning()
        self.save_classes()
        #print("Classes are saved")


    def preprocess_image(imgs):
        imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, 3), dtype=np.uint8)
        for i in range(imgs.shape[0]):
            #imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)
            img = clahe_augment(img)
            img = contrast_augment(img)
            img = postprocessing(img, self.imageShape)
            imgs_p[i] = img

        imgs_p = imgs_p[..., np.newaxis]
        return imgs_p

    def getData(self, ):

        def preprocess_image(imgs):
            imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, 3), dtype=np.uint8)
            for i in range(imgs.shape[0]):
                #imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)
                img = clahe_augment(img)
                img = contrast_augment(img)
                img = postprocessing(img, self.imageShape)
                imgs_p[i] = img
            imgs_p = imgs_p[..., np.newaxis]
            return imgs_p

        if self.gender == 'M':
            X_train_m, y_train_m = load_train_data('M')
            X_train_m = preprocess_image(X_train_m)
            X_train_m = X_train_m.astype('float32')/255

            print("Train samples (male): {}".format(X_train_m.shape))
            return X_train_m, y_train_m

        if self.gender == 'F':
            X_train_f, y_train_f = load_train_data('F')
            X_train_f = preprocess_image(X_train_f)
            X_train_f = X_train_f.astype('float32')/255

            print("Train samples (female): {}".format(X_train_f.shape))
            return X_train_f, y_train_f

        if self.gender == 'ALL':
            X_train_all, y_train_all = load_train_data('ALL')
            X_train_all = preprocess_image(X_train_all)
            X_train_all = X_train_all.astype('float32')/255

            print("Train samples (All): {}".format(X_train_all.shape))
            return X_train_all, y_train_all

    def load(self):
        print("Creating model")
        self.load_classes()
        self._create()
        self.model.load_weights(config.get_fine_tuned_weights_path())
        return self.model

    @staticmethod
    def save_classes():
        joblib.dump(config.classes, config.get_classes_path())

    def get_input_tensor(self):
        if util.get_keras_backend_name() == 'theano':
            return Input(shape=(3,) + self.img_size)
        else:
            return Input(shape=self.img_size + (3,))

    def get_input_shape(self):
        X_train_m, y_train_m, X_valid_m, y_valid_m, X_test_m, y_test_m = self.getData()

        return X_train_m[0].shape

    @staticmethod
    def make_net_layers_non_trainable(model):
        for layer in model.layers:
            layer.trainable = False

    def freeze_top_layers(self):
        if self.freeze_layers_number:
            print("Freezing {} layers".format(self.freeze_layers_number))
            for layer in self.model.layers[:self.freeze_layers_number]:
                layer.trainable = False
            for layer in self.model.layers[self.freeze_layers_number:]:
                layer.trainable = True


    @staticmethod
    def get_callbacks(weights_path, monitor='val_loss'):
        model_checkpoint = ModelCheckpoint(weights_path, save_best_only=True, save_weights_only=True, monitor=monitor)
        csv_logger = CSVLogger('log_resnetoriginalcv.csv', append=True, separator=',')
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
                                  write_grads=False,
                                  write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                  embeddings_metadata=None)
        reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
        stopping = EarlyStopping (monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
        return [model_checkpoint, csv_logger, tensorboard, reduceLR, stopping]

    @staticmethod
    def apply_mean(image_data_generator):
        """Subtracts the dataset mean"""
        image_data_generator.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))

    @staticmethod
    def load_classes():
        config.classes = joblib.load(config.get_classes_path())

    def load_img(self, img_path):
        img = image.load_img(img_path, target_size=self.img_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return preprocess_input(x)[0]


    @staticmethod
    def huber_loss(y_true, y_pred):
        # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
        # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
        # for details.
        clip_value = 1
        assert clip_value > 0.

        x = y_true - y_pred
        if np.isinf(clip_value):
            # Spacial case for infinity since Tensorflow does have problems
            # if we compare `K.abs(x) < np.inf`.
            return .5 * K.square(x)

        condition = K.abs(x) < clip_value
        squared_loss = .5 * K.square(x)
        linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
        if K.backend() == 'tensorflow':
            import tensorflow as tf
            if hasattr(tf, 'select'):
                return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
            else:
                return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
        elif K.backend() == 'theano':
            from theano import tensor as T
            return T.switch(condition, squared_loss, linear_loss)
        else:
            raise RuntimeError('Unknown backend "{}".'.format(K.backend()))


    def get_train_datagen(self, X_train, y_train):
        idg = ImageDataGenerator(rotation_range=30, height_shift_range=0.01, horizontal_flip=True)
        self.apply_mean(idg)
        return idg.flow(X_train, y_train, batch_size=self.batch_size)

    def get_validation_datagen(self, X_test, y_test):
        idg = ImageDataGenerator()
        self.apply_mean(idg)
        return idg.flow(X_test, y_test, batch_size=self.batch_size)





