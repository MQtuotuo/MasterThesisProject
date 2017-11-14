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
        #self.model.compile(loss='mean_squared_error', metrics=['mae'], optimizer=Adam(lr=1.0e-5))
        #self.model.compile(loss=self.huber_loss, metrics=['mae'], optimizer='adam')
        #self.model.load_weights(config.get_fine_tuned_weights_path())

        #X_train, y_train, X_test, y_test = self.getData()
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
            CVscores.append(score)
            print("validation MSE: ", score)
        CVmean = np.mean(CVscores)
        CVstd = np.std(CVscores)
        CVpct = 100 * CVstd / CVmean
        print("For " + str(n_fold) + "-fold CV, avg MSE: " + format(CVmean, '.4f') +", std dev of MSE: " + format(CVstd, '.4f') +" (" + format(CVpct,'.2f') + "%)")

        '''
        history = self.model.fit(
                X_train, y_train,
                batch_size = self.batch_size,
                epochs=self.nb_epoch,
                validation_data=(X_test, y_test),
                #validation_data=(X_valid, y_valid),
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
        '''


    def train(self):
        print("Creating model...")
        self._create()
        print("Model is created")
        print("Fine tuning...")
        self._fine_tuning()
        self.save_classes()
        #print("Classes are saved")



    def getData(self, ):

        PATH = '/home/ming/workspace.old/RSNA.boneage'
        # Define data path
        data_path = PATH + '/train'
        # print(data_path)
        img_path = '/home/ming/workspace.old/unet/output'

        dataframe = pandas.read_csv(os.path.join(PATH, 'train.csv'), usecols=[0, 1, 2])
        #dataframe_test = pandas.read_csv(os.path.join(data_path, 'test.csv'), usecols=[0, 1])
        def maleToInt(male):
            if str(male) == 'True':
                return 1
            else:
                return 0

        def sexToInt(sex):
            if str(sex) == 'M':
                return 1
            else:
                return 0


        dataframe.male = dataframe.male.apply(maleToInt)
        #dataframe_test.sex = dataframe_test.sex.apply(sexToInt)
        print(dataframe.head())
        #print(dataframe_test.head())

        def preprocessing (x, y = None, resizeTo = (224,224)):
            # resize to intermediate size
            x = resizeToFit (x, resizeTo)
            x = pseudoRGB (x, "clahe")/255
            if y is not None:
                y = resizeToFit (y, resizeTo)
                y = y.astype('float32')/255
                return x, y
            return x

        def postprocessing (  x, y = None, method = "crop", visualize = False):
            imageShape = (224, 224, 3)
            if method == "resize":
                x = resizeToFit (x, imageShape)
                if y is not None:
                    y = resizeToFit (y, imageShape)
                    y = y.astype('float32')/255
                    return x, y
                return x
                
            if method == "crop":
                offsetRow = (x.shape[0] - imageShape[0])//2
                offsetCol = (x.shape[1] - imageShape[1])//2
                x = x[offsetRow:offsetRow+imageShape[0], offsetCol:offsetCol + imageShape[1], :].copy()

                if visualize == True:
                    fig2 = plt.figure(figsize = (10,5)) # create a 5 x 5 figure 
                    for i in range(0, 9):
                        pyplot.subplot(1, 2,  1)
                        pyplot.imshow(x)
                    pyplot.show()

                if y is not None:
                    y = resizeToFit (y, imageShape)
                    y = y.astype('float32')/255
                    return x, y
                return x       
              
        def clahe_augment(img):
            clahe_low = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
            clahe_medium = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            clahe_high = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
            img_low = clahe_low.apply(img)
            img_medium = clahe_medium.apply(img)
            img_high = clahe_high.apply(img)
            augmented_img = np.array([img_low, img_medium, img_high])
            augmented_img = np.swapaxes(augmented_img,0,1)
            augmented_img = np.swapaxes(augmented_img,1,2)
            return augmented_img

        def contrast_augment(img):
            clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
            l, a, b = cv2.split(lab)  # split on 3 different channels
            l2 = clahe.apply(l)  # apply CLAHE to the L-channel
            lab = cv2.merge((l2,a,b))  # merge channels
            augmented_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
            return augmented_img



        img_resize = (512, 512)
        x = int((img_resize[0] - self.img_size[0]) / 2)
        y = int((img_resize[1] - self.img_size[1]) / 2)
        w = int(x + self.img_size[0])
        h = int(y + self.img_size[1])

        img_male_list = []
        img_female_list = []
        y_male = []
        y_female = []
        img_all_list = []
        y_all = []

        if self.gender == 'M':
            for index, row in dataframe.iterrows():
                if row['male'] == 1:  # all the male
                    filepath_temp = os.path.join(data_path + '/' + str(row['id']) + '.png')
                    image = cv2.imread(filepath_temp, 0)
                    
                    img = resizeToFit(image, (512, 512))
                    img = clahe_augment(img)
                    img = contrast_augment(img)
                    img = postprocessing(img)
                    #cropped = img[x:w, y:h]      
                    #img = preprocessing(image)
                    img_male_list.append(img)
                    y_male.append(row['boneage'])


            img_male_data = np.array(img_male_list)
            img_male_data = img_male_data.astype('float32')
            img_male_data /=255

            y_male = np.array(y_male)
            y_male = y_male.astype('float32')
            x_m, y_m = shuffle(img_male_data, y_male, random_state=2)
            X_train_m = x_m
            y_train_m = y_m

            # Split the dataset (male)
            #X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(x_m, y_m, test_size=0.2, random_state=2)   
            #X_valid_m, X_test_m, y_valid_m, y_test_m = train_test_split(X_test_m, y_test_m, test_size=0.5, random_state=2)
            print("Train samples (male): {}".format(X_train_m.shape))
            #print("Validation samples (male): {}".format(X_valid_m.shape))
            #print("Test samples (male): {}".format(X_test_m.shape))
            return X_train_m, y_train_m

        if self.gender == 'F':
            for index, row in dataframe.iterrows():
                if row['male'] == 0:  # all the female
                    filepath_temp = os.path.join(img_path + '/' + str(row['id']) + '_new.png')
                    image = cv2.imread(filepath_temp, -1)
                    image = cv2.resize(image, img_resize)
                    cropped = image[x:w, y:h]
                    #blur = cv2.GaussianBlur(cropped,(5,5),0)
                    #input_img = pseudoRGB(blur, "clahe", visualize=False)
                    img_female_list.append(cropped)
                    y_female.append(row['boneage'])
            img_female_data = np.array(img_female_list)
            img_female_data = img_female_data.astype('float32')
            img_female_data /= 255

            y_female = np.array(y_female)
            y_female = y_female.astype('float32')

            x_f, y_f = shuffle(img_female_data, y_female, random_state=2)

            # Split the dataset (female)
            X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(x_f, y_f, test_size=0.2, random_state=2)
           
            print("Train samples (female): {}".format(X_train_f.shape))
            print("Test samples (female): {}".format(X_test_f.shape))
            return X_train_f, y_train_f, X_test_f, y_test_f

        if self.gender == 'ALL':
            for index, row in dataframe.iterrows():
                filepath_temp = os.path.join(img_path + '/' + str(row['id']) + '.png')
                image = cv2.imread(filepath_temp, -1)
                image = cv2.resize(image, img_resize)
                cropped = image[x:w, y:h]
                blur = cv2.GaussianBlur(cropped,(5,5),0)
                input_img = pseudoRGB(blur, "clahe", visualize=False)
                img_all_list.append(input_img)
                y_all.append(row['boneage'])
            img_all_data = np.array(img_all_list)
            img_all_data = img_all_data.astype('float32')
            img_all_data /= 255

            y_all = np.array(y_all)
            y_all = y_all.astype('float32')

            x_all, y_all = shuffle(img_all_data, y_all, random_state=2)

            # Split the dataset (female)
            X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(x_all, y_all, test_size=0.2, random_state=2)
        
            print("Train samples (all): {}".format(X_train_all.shape))
            print("Test samples (all): {}".format(X_test_all.shape))
            return X_train_all, y_train_all, X_test_all, y_test_all

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





