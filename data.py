from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread
from shutil import copyfile, rmtree
import cv2
from pseudoRGB import pseudoRGB
from resizeToFit import *
import keras.backend as K

data_path = '/home/ming/workspace.old'
img_path = os.path.join(data_path, 'RSNA.boneage', 'train')
image_rows = 512
image_cols = 512
dim = 3
img_list = []
y = []


def maleToInt(male):
            if str(male) == 'True':
                return 1
            else:
                return 0

def postprocessing (x, y = None, method = "crop", visualize = False):
        imageShape = (224, 224, 3)
        if method == "resize":
            x = resizeToFit (x, imageShape)
            if y is not None:
                y = resizeToFit (y, imageShape)
                y = y.astype('float32')/255
                return x, y
            return x
        
        if method == "crop":
            # assume quadratic images
            #print (x.shape)
            offsetRow = (x.shape[0] - imageShape[0])//2
            offsetCol = (x.shape[1] - imageShape[1])//2
            #print (offsetRow )
            #print (offsetCol )
            x = x[offsetRow:offsetRow+imageShape[0], offsetCol:offsetCol + imageShape[1], :].copy()
            #print (x.shape)
            # do simple cropping 

            if visualize == True:
                fig2 = plt.figure(figsize = (10,5)) # create a 5 x 5 figure 
                for i in range(0, 9):
                    pyplot.subplot(1, 2,  1)
                    pyplot.imshow(x)
                    #pyplot.subplot(1, 2, 2)
                    #pyplot.imshow(y)
                    # show the plot
                pyplot.show()
                
            
            if y is not None:
                y = resizeToFit (y, imageShape)
                y = y.astype('float32')/255
                return x, y
            return x



def create_train_data(gender):
    train_data_path = os.path.join(data_path, 'RSNA.boneage', 'train')
    dataframe = pandas.read_csv(os.path.join(data_path, 'RSNA.boneage', 'train.csv'), usecols=[0, 1, 2])
    dataframe.male = dataframe.male.apply(maleToInt)
    print(dataframe.head())

    i = 0
    print('-'*30)
    print('Creating train images...')
    print('-'*30)
    imageShape = (image_rows, image_cols, dim)

    if gender == 'M':
        for index, row in dataframe.iterrows():
            if row['male'] == 1:  # all the male
                filepath_temp = os.path.join(img_path + '/' + str(row['id']) + '.png')
                image = cv2.imread(filepath_temp, -1)
                img = resizeToFit(image, imageShape)
                img_list.append(img)
                y.append(row['boneage'])

                if i % 100 == 0:
                    print('Done: {0} images'.format(i))
                    i += 1

        img_data = np.array(img_list)
        img_data = img_data.astype('float32')

        y = np.array(y)
        y = y.astype('float32')

        x_m, y_m = shuffle(img_data, y, random_state=2)

        # Split the dataset (male)
        X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(x_m, y_m, test_size=0.2, random_state=2)
        #X_valid_m, X_test_m, y_valid_m, y_test_m = train_test_split(X_test_m, y_test_m, test_size=0.5, random_state=2)
       
        print('Loading done.')
        print("Train samples (male): {}".format(X_train_m.shape))
        print("Test samples (male): {}".format(X_test_m.shape))
        np.save('X_train_m.npy', X_train_m)
        np.save('y_train_m.npy', y_train_m)
        np.save('X_test_m.npy', X_test_m)
        np.save('y_test_m.npy', y_test_m)
        print('Saving to .npy files done.')

    if gender == 'F':
        for index, row in dataframe.iterrows():
            if row['male'] == 0:  # all the male
                filepath_temp = os.path.join(img_path + '/' + str(row['id']) + '.png')
                image = cv2.imread(filepath_temp, -1)
                img = resizeToFit(image, imageShape)
                img_list.append(img)
                y.append(row['boneage'])

                if i % 100 == 0:
                    print('Done: {0} images'.format(i))
                    i += 1

        img_data = np.array(img_list)
        img_data = img_data.astype('float32')

        y = np.array(y)
        y = y.astype('float32')
        x_f, y_f = shuffle(img_data, y, random_state=2)

        # Split the dataset (male)
        X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(x_f, y_f, test_size=0.2, random_state=2)
       
        print('Loading done.')
        print("Train samples (female): {}".format(X_train_f.shape))
        print("Test samples (female): {}".format(X_test_f.shape))
        np.save('X_train_f.npy', X_train_f)
        np.save('y_train_f.npy', y_train_f)
        np.save('X_test_f.npy', X_test_f)
        np.save('y_test_f.npy', y_test_f)
        print('Saving to .npy files done.')

    if gender == 'ALL':
        for index, row in dataframe.iterrows():
           
            filepath_temp = os.path.join(img_path + '/' + str(row['id']) + '.png')
            image = cv2.imread(filepath_temp, -1)
            img = resizeToFit(image, imageShape)
            img_list.append(img)
            y.append(row['boneage'])

            if i % 100 == 0:
                print('Done: {0} images'.format(i))
                i += 1

        img_data = np.array(img_list)
        img_data = img_data.astype('float32')

        y = np.array(y)
        y = y.astype('float32')

        x_all, y_all = shuffle(img_data, y, random_state=2)

        # Split the dataset (male)
        X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(x_all, y_all, test_size=0.2, random_state=2)
       
        print('Loading done.')
        print("Train samples (all): {}".format(X_train_all.shape))
        print("Test samples (all): {}".format(X_test_all.shape))
        np.save('X_train_all.npy', X_train_all)
        np.save('y_train_all.npy', y_train_all)
        np.save('X_test_all.npy', X_test_all)
        np.save('y_test_all.npy', y_test_all)
        print('Saving to .npy files done.')

       
       

def load_train_data(gender):
    if gender = "M":
        X_train = np.load('X_train_m.npy')
        y_train = np.load('y_train_m.npy')
        X_test = np.load('X_test_m.npy')
        y_test = np.load('y_test_m.npy')
        return X_train, y_train, X_test, y_test
    if gender = "F":
        X_train = np.load('X_train_f.npy')
        y_train = np.load('y_train_f.npy')
        X_test = np.load('X_test_f.npy')
        y_test = np.load('y_test_f.npy')
        return X_train, y_train, X_test, y_test
    if gender = "ALL":
        X_train = np.load('X_train_all.npy')
        y_train = np.load('y_train_all.npy')
        X_test = np.load('X_test_m.npy')
        y_test = np.load('y_test_m.npy')
        return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    create_train_data('M')
    #load_train_data('M')





