from __future__ import print_function

import os
import numpy as np
import pandas
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
dim = 1
npy_path = '/home/ming/workspace.old/Bone/npy'



def maleToInt(male):
            if str(male) == 'True':
                return 1
            else:
                return 0



def create_train_data(gender):
    train_data_path = os.path.join(data_path, 'RSNA.boneage', 'train')
    dataframe = pandas.read_csv(os.path.join(data_path, 'RSNA.boneage', 'train.csv'), usecols=[0, 1, 2])
    dataframe.male = dataframe.male.apply(maleToInt)
    print(dataframe.head())

    i = 0
    img_list = []
    y = []
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

        X_train_m = np.array(img_list)
        #X_train_m = X_train_m.astype('float32')

        y_train_m = np.array(y)
        y_train_m = y_train_m.astype('float32')
       
        print('Loading done.')
        print("Train samples (male): {}".format(X_train_m.shape))
        np.save(os.path.join(npy_path, 'X_train_m.npy'), X_train_m)
        np.save(os.path.join(npy_path, 'y_train_m.npy'), y_train_m)
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

        X_train_f = np.array(img_list)
        #X_train_f = X_train_f.astype('float32')

        y_train_f = np.array(y)
        y_train_f = y_train_f.astype('float32')
        
        print('Loading done.')
        print("Train samples (female): {}".format(X_train_f.shape))
        np.save(os.path.join(npy_path, 'X_train_f.npy'), X_train_f)
        np.save(os.path.join(npy_path, 'y_train_f.npy'), y_train_f)
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

        X_train_all = np.array(img_list)
        #X_train_all = X_train_all.astype('float32')

        y_train_all = np.array(y)
        y_train_all = y_train_all.astype('float32')

        
        print('Loading done.')
        print("Train samples (all): {}".format(X_train_all.shape))
        np.save(os.path.join(npy_path, 'X_train_all.npy'), X_train_all)
        np.save(os.path.join(npy_path, 'y_train_all.npy'), y_train_all)
        print('Saving to .npy files done.')

       
       

def load_train_data(gender):
    if gender == "M":
        X_train = np.load(os.path.join(npy_path, 'X_train_m.npy'))
        y_train = np.load(os.path.join(npy_path, 'y_train_m.npy'))
        return X_train, y_train
    if gender == "F":
        X_train = np.load(os.path.join(npy_path, 'X_train_f.npy'))
        y_train = np.load(os.path.join(npy_path, 'y_train_f.npy'))
        return X_train, y_train
    if gender == "ALL":
        X_train = np.load(os.path.join(npy_path, 'X_train_all.npy'))
        y_train = np.load(os.path.join(npy_path, 'y_train_all.npy'))
        return X_train, y_train
if __name__ == '__main__':
    create_train_data('M')
    #load_train_data('M')





