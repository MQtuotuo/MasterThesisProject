
from keras import applications, Input
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import os
import cv2
from pseudoRGB import pseudoRGB
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
# path to the model weights files.
weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 32
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

# build the VGG16 network
input_tensor = Input(shape=(224, 224, 3))
base_model = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dropout(0.5))
top_model.add(Dense(2048, activation='elu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
#model.add(top_model)
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will ånot be updated)
for layer in model.layers[:3]:
    layer.trainable = True
for layer in model.layers[150:]:
    layer.trainable = True
for layer in model.layers[3:150]:
    layer.trainable = False


# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='mean_squared_error', metrics=['mae'], optimizer=Adam(lr=1.0e-5))



model.summary()
from keras.callbacks import TensorBoard, CSVLogger

csv_logger = CSVLogger('log_second.csv', append=True, separator=',')

X_train, y_train, X_valid, y_valid, X_test, y_test = getData('M')
# fine-tune the model
history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_valid, y_valid),
                    callbacks=[csv_logger])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
import matplotlib.pyplot as plt

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')

plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.close()

score = model.evaluate(X_valid)

scores = model.predict(X_valid)

correct = 0

print("Loss: ", score[0], "MAE: ", score[1])
