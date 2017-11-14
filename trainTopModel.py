
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import os
import cv2
from pseudoRGB import pseudoRGB
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from time import time

# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'bottleneck_fc_model.h5'
epochs = 100
batch_size = 32
from keras.callbacks import TensorBoard, CSVLogger
import pandas
PATH = os.getcwd()
def getData(gender):
    PATH = os.getcwd()
    # Define data path
    data_path = PATH + '/RSNA'
    # print(data_path)
    img_path = '/data/RSNA.boneage/images'
    img_size = (224, 224)
    dataframe = pandas.read_csv(os.path.join(data_path, 'train.csv'), usecols=[0, 1, 2])

    print(dataframe.head())

    def maleToInt(male):
        if str(male) == 'True':
            return 1
        else:
            return 0

    dataframe.male = dataframe.male.apply(maleToInt)
    print(dataframe.head())

    img_resize = (512, 512)
    x = int((img_resize[0] - img_size[0]) / 2)
    y = int((img_resize[1] - img_size[1]) / 2)
    w = int(x + img_size[0])
    h = int(y + img_size[1])

    def to_rgb3a(im):
        return np.dstack([im.astype(np.float32)] * 3)

    img_male_list = []
    img_female_list = []
    y_male = []
    y_female = []
    img_all_list = []
    y_all = []

    if gender == 'M':
        for index, row in dataframe.iterrows():
            if row['male'] == 1:  # all the male
                filepath_temp = os.path.join(img_path + '/' + str(row['id']) + '.png')
                image = cv2.imread(filepath_temp, -1)
                image = cv2.resize(image, img_resize)
                cropped = image[x:w, y:h]
                input_img = pseudoRGB(cropped, "clahe", visualize=False)
                img_male_list.append(input_img)
                y_male.append(row['boneage'])

        img_male_data = np.array(img_male_list)
        img_male_data = img_male_data.astype('float32')
        img_male_data /= 255
        y_male = np.array(y_male)
        y_male = y_male.astype('float32')
        x_m, y_m = shuffle(img_male_data, y_male, random_state=2)

        # Split the dataset (male)
        X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(x_m, y_m, test_size=0.2, random_state=2)
        X_valid_m, X_test_m, y_valid_m, y_test_m = train_test_split(X_test_m, y_test_m, test_size=0.5,
                                                                    random_state=2)
        print("Train samples (male): {}".format(X_train_m.shape))
        print("Validation samples (male): {}".format(X_valid_m.shape))
        print("Test samples (male): {}".format(X_test_m.shape))
        return X_train_m, y_train_m, X_valid_m, y_valid_m, X_test_m, y_test_m

    if gender == 'F':
        for index, row in dataframe.iterrows():
            if row['male'] == 0:  # all the female
                filepath_temp = os.path.join(img_path + '/' + str(row['id']) + '.png')
                image = cv2.imread(filepath_temp, -1)
                image = cv2.resize(image, img_resize)
                cropped = image[x:w, y:h]
                input_img = pseudoRGB(cropped, "clahe", visualize=False)
                img_female_list.append(input_img)
                y_female.append(row['boneage'])
        img_female_data = np.array(img_female_list)
        img_female_data = img_female_data.astype('float32')
        img_female_data /= 255

        y_female = np.array(y_female)
        y_female = y_female.astype('float32')

        x_f, y_f = shuffle(img_female_data, y_female, random_state=2)

        # Split the dataset (female)
        X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(x_f, y_f, test_size=0.2, random_state=2)
        X_valid_f, X_test_f, y_valid_f, y_test_f = train_test_split(X_test_f, y_test_f, test_size=0.5,
                                                                    random_state=2)
        print("Train samples (female): {}".format(X_train_f.shape))
        print("Validation samples (female): {}".format(X_valid_f.shape))
        print("Test samples (female): {}".format(X_test_f.shape))
        return X_train_f, y_train_f, X_valid_f, y_valid_f, X_test_f, y_test_f

    if gender == 'ALL':
        for index, row in dataframe.iterrows():
            filepath_temp = os.path.join(img_path + '/' + str(row['id']) + '.png')
            image = cv2.imread(filepath_temp, -1)
            image = cv2.resize(image, img_resize)
            cropped = image[x:w, y:h]
            input_img = pseudoRGB(cropped, "clahe", visualize=False)
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
        X_valid_all, X_test_all, y_valid_all, y_test_all = train_test_split(X_test_all, y_test_all, test_size=0.5,
                                                                            random_state=2)
        print("Train samples (all): {}".format(X_train_all.shape))
        print("Validation samples (all): {}".format(X_valid_all.shape))
        print("Test samples (all): {}".format(X_test_all.shape))
        return X_train_all, y_train_all, X_valid_all, y_valid_all, X_test_all, y_test_all

def save_bottlebeck_features(gender):

    # build the VGG16 network
    model = applications.ResNet50(include_top=False, weights='imagenet')
    X_train, y_train, X_valid, y_valid, X_test, y_test = getData(gender)

    bottleneck_features_train = model.predict(
        X_train, 32)
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    bottleneck_features_validation = model.predict(
        X_valid, 32)

    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)


def train_top_model(gender):
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    X_train, y_train, X_valid, y_valid, X_test, y_test = getData(gender)

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    model = Sequential()

    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', metrics=['mae'], optimizer='adam')

    model.summary()
    csv_logger = CSVLogger('log.csv', append=True, separator=',')
    # 	tensorboard --logdir=logs/
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                              write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                              embeddings_metadata=None)

    history = model.fit(train_data, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, y_valid),
              callbacks = [tensorboard, csv_logger])

    model.save_weights(top_model_weights_path)
    # list all data in history
    import matplotlib.pyplot as plt
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('model mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(PATH % 'mae')
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(PATH % 'loss')
    plt.close()

save_bottlebeck_features('M')
train_top_model('M')
