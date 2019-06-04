
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import os
import timeit
from matplotlib.patches import ConnectionPatch
from pprint import pprint
from sklearn.metrics import average_precision_score
from random import sample

from own_implementations import HarrisMataHarris, ShiTomasi, CensureClass, RootSIFT
from const import *

TIME_MEASURES_TXT = 'time_measures_zver_3.txt'


def alreadyCompiledKeypoints(dataset_path):
    '''
    Check which detectors and descriptors you already compiled.
    '''

    if 'kp.npz' in os.listdir(dataset_path + '/v_london'):
        file = np.load(dataset_path + '/v_london/kp.npz')
        print('Compiled keypoints: {}'.format(file.files))
    else:
        print('There are no keypoints already compiled in kp.npz')

    if 'des.npz' in os.listdir(dataset_path + '/v_london'):
        file = np.load(dataset_path + '/v_london/des.npz')
        print('Compiled descriptors: {}'.format(file.files))
    else:
        print('There are no descriptors already compiled in des.npz')


def deleteAllKeypoints(dataset_path):
    '''
    Delete all kp.npz and des.npz in all folders.
    '''
    folders = glob.glob(dataset_path)

    for folder in folders:
        desss = glob.glob(folder + '/des*')
        kpppp = glob.glob(folder + '/kp*')
        for d in desss:
            os.remove(d)
        for k in kpppp:
            os.remove(k)


def createKeypoints(detector_name, descriptor_name, dataset_path, all_at_once=False):
    '''
    For a given detector and descriptor pair, save keypoints
    and descriptors to kp.npz and des.npz in every folder for whole sequence
    '''

    sequence_images = {}
    folders = glob.glob(dataset_path)

    for folder in folders:
        folder_name = folder.split('/')[-1]
        print('Working on folder {}'.format(folder_name))

        # load sequence images in list
        sequence_images[folder_name] = glob.glob(folder + '/*.ppm')
        sequence_images[folder_name] = sorted(sequence_images[folder_name])
        sequence_images[folder_name] = [cv2.imread(im) for im in sequence_images[folder_name]]

        # convert images to gray
        images_ = [ cv2.cvtColor(im ,cv2.COLOR_RGB2GRAY) for im in sequence_images[folder_name]]

        # create detector and descriptor instances
        detector = [ all_detectors[detector_name]() for im in sequence_images[folder_name]]
        descriptor = [ all_descriptors[descriptor_name]() for im in sequence_images[folder_name]]

        kp = []
        des = []

        for id1, algorithm in enumerate(zip(detector,descriptor)):

            if not all_at_once:
                start_time = timeit.default_timer()
                kp_ = algorithm[0].detect(images_[id1],None)

                kp_, des_ = algorithm[1].compute(images_[id1], kp_)
                end_time = timeit.default_timer()

                if (des_ == None).any():
                    des_ = np.array([])

                with open(TIME_MEASURES_TXT,'a+') as file:
                    file.write('{},{},{}\n'.format(detector_name,
                                                   descriptor_name,
                                                   end_time-start_time))
            else:
                start_detector_and_descriptor = timeit.default_timer()
                kp_, des_ = det.detectAndCompute(images_[id1],None)
                end_detector_and_descriptor = timeit.default_timer()

                if (des_ == None).any():
                    des_ = np.array([])

                with open(TIME_MEASURES_TXT,'a+') as file:
                    file.write('{},{},{}\n'.format(detector_name, descriptor_name,
                        end_detector_and_descriptor-start_detector_and_descriptor))

            kp_np = np.array([(k.pt[0], k.pt[1], k.angle, k.size, k.response) for k in kp_])

            kp.append(kp_np)
            des.append(des_)

        if 'kp.npz' in os.listdir(folder):
            file = np.load(folder + '/kp.npz')
            elements = dict(file)
            elements[detector_name] = kp
            np.savez(folder + '/kp.npz', **elements)

        else:
            np.savez(folder + '/kp.npz', **{detector_name:kp})

        if 'des.npz' in os.listdir(folder):
            file = np.load(folder + '/des.npz')
            elements = dict(file)
            elements[detector_name + '_' + descriptor_name] = des
            np.savez(folder + '/des.npz', **elements)

        else:
            nm = detector_name + '_' + descriptor_name
            np.savez(folder + '/des.npz', **{nm:des})


def getTransformations(dataset_path):
    transformations = {}
    folders = glob.glob(dataset_path)

    def load_transform(tr):
        with open(tr) as file:
            s = file.read()
            nrs = re.split('\n| ',s)[:-1]
            nrs = [nr for nr in nrs if nr != '']
            return np.array(nrs).reshape(3,3).astype(np.float)

    for folder in folders:
        folder_name = folder.split('/')[-1]
        transformations[folder_name] = glob.glob(folder + '/H*')
        transformations[folder_name] = sorted(transformations[folder_name])
        transformations[folder_name] = [load_transform(tr) for tr in transformations[folder_name]]

    return transformations


def removeUncommonPoints(detector_name, descriptor_name, dataset_path):
    '''
    Remove keypoints from ref image that do not appear on sequence images
    when imaged with homography H.
    This function expects that keypoints already exist in kp.npz and des.npz
    for given detector and descriptor names.
    '''
    transformations = getTransformations(dataset_path)
    folders = glob.glob(dataset_path)


    for folder in folders:
        folder_name = folder.split('/')[-1]
        print('#########################################')
        print('Working on folder {}'.format(folder_name))

<<<<<<< HEAD
        # load keypoints
        kp_file = np.load(folder + '/kp.npz')
=======
        kp_file = np.load(folder + '/kp.npz', allow_pickle=True)
>>>>>>> edc09c5e1670bff8772de4f640622a18657d906c
        kp = kp_file[detector_name]
        kp = list(kp)

        des_file = np.load(folder + '/des.npz', allow_pickle=True)
        nm = detector_name + '_' + descriptor_name
        des = des_file[nm]
        des = list(des)

        indexes_to_remove = []

        # remove keypoints from ref image that do not appear on sequence images
        remove = set()
        kp_ = kp[0].copy()

        # iterate over homographies H and image points onto sequence image
        for id2, tr in enumerate(transformations[folder_name]):
            # image points
            if kp_.shape[0] == 0:
                with open('missing_kp_on_ref','a+') as fl:
                    fl.write('{}: algo with no kp on 1.ppm for folder {} and image {}.'.format(detector_name,
                                                                                               folder_name,
                                                                                               id2+2))
                print('THERE ARE NO KEYPOINTS ON IMAGE 1.ppm IN FOLDER {}'.format(folder_name))
                continue
            points = np.c_[ kp_[:,[0,1]] , np.ones(kp_.shape[0])]
            imaged_points = np.dot(tr, points.T)
            imaged_points_normal = imaged_points/imaged_points[2,:]

            # get bounds of image on which we are projecting
            image_size = cv2.imread(folder + '/' + str(id2+2) + '.ppm' )
            image_size = image_size.shape

            # get indexes that are out of bounds on the sequence image
            x_indexes_out_of_bounds = np.where((imaged_points_normal[0,:] < 0) |
                                               (image_size[1] < imaged_points_normal[0,:]))[0]


            y_indexes_out_of_bounds = np.where((imaged_points_normal[1,:] < 0) |
                                               (image_size[0] < imaged_points_normal[1,:]))[0]

            # add the indexes to set
            remove = remove.union(x_indexes_out_of_bounds)
            remove = remove.union(y_indexes_out_of_bounds)

        # create a list from the set
        indexes_to_remove = list(remove)

        # delete the indexes
        print('Removing {} keypoints from image {}/1.ppm'.format(len(indexes_to_remove),folder_name))
        print('old size: {}'.format(kp[0].shape))
        kp[0] = np.delete(kp[0], indexes_to_remove, 0)
        print('new size: {}'.format(kp[0].shape))

        des[0] = np.delete(des[0], indexes_to_remove, 0)

        # save the new keypoints to disk
        elements = dict(kp_file)
        elements[detector_name] = kp
        np.savez(folder + '/kp.npz', **elements)

        elements = dict(des_file)
        elements[nm] = des
        np.savez(folder + '/des.npz', **elements)


def createMeTable():
    '''
    Loads execution times for detectors+descriptors and prints out how the
    table should look in latex.
    '''
    import pandas as pd

    data = pd.read_csv('rezultati/time_measures_zver_1.txt',
                       sep=",",
                       header=None,
                       names=['det','des','time'],
                       index_col=None)
    data1 = pd.read_csv('rezultati/time_measures_lfnet_bartools.txt',
                       sep=",",
                       header=None,
                       names=['det','des','time'],
                       index_col=None)

    data2 = pd.read_csv('rezultati/time_measures_superpoint_bartools.txt',
                       sep=",",
                       header=None,
                       names=['det','des','time'],
                       index_col=None)
    data = data.append(data1, ignore_index=True)
    data = data.append(data2, ignore_index=True)

    data['time'] = data['time'].astype('float64')

    data2 = pd.DataFrame(columns=['det','des','time'])
    for name, df in data.groupby(['det','des']):
        me = np.mean(df['time'])
        data2 = data2.append(pd.Series([name[0].upper(),name[1].upper(),me],
                                        index=['det','des','time']),
                             ignore_index=True)

    data2.sort_values(by=['time','det'], inplace=True)
    data2['time'] = data2['time'].round(3)

    data2.reset_index(inplace=True,drop=True)

    print(data2.head())

    tt = min(data2.shape[0],10)
    data2 = data2.loc[:tt,:]

    print('Detector & ' + ' & '.join(data2['det']) + ' \\\\')
    print('\\hline')
    print('Descriptor & ' + ' & '.join(data2['des']) + ' \\\\')
    print('\\hline')
    print('Time & ' + ' & '.join(data2['time'].astype('str')) + ' \\\\')
    print('\\hline')


def read_keypoints(folder, detector_name, descriptor_name):
    # Get keypoints for the sequence
    kp_all = np.load(folder + '/kp.npz', allow_pickle=True)
    kp = kp_all[detector_name]

    # Get descriptors for the sequence
    des_all = np.load(folder + '/des.npz', allow_pickle=True)
    nm = detector_name + '_' + descriptor_name
    des = des_all[nm]

    return list(kp), list(des)



def read_next_keypoints(detector_name, descriptor_name, folders, folder_id, m=100):

    kp_all = []
    des_all = []

    random_images = sample(range(len(folders)), m)

    for ind in random_images:
        seq = (folder_id+ind) % len(folders)
        image_nr = ind%6

        kp = np.load(folders[seq] + '/kp.npz')
        kp = list(kp[detector_name])
        kp = kp[image_nr]
        kp_all.append(kp)

        des = np.load(folders[seq]+'/des.npz')
        nm = detector_name + '_' + descriptor_name
        des = list(des[nm])
        des = des[image_nr]
        des_all.append(des)

    return kp_all, des_all

if __name__ == '__main__':
    import argparse

    parser_of_args = argparse.ArgumentParser(description='Create kp and des')
    parser_of_args.add_argument('detector_name', type=str,
                                help='name of the detector')
    parser_of_args.add_argument('descriptor_name', type=str,
                                help='name of descriptor')
    parser_of_args.add_argument('dataset_path', type=str,
                                help='path to hpatches dataset')

    args = parser_of_args.parse_args()

    # project_root = '/home/davidboja/PycharmProjects/FER/hpatches-benchmark/python/ISPA'
    # dataset_path = project_root + '/hpatches-sequences-release/*'

    dataset_path = args.dataset_path + '/*'

    createKeypoints(args.detector_name, args.descriptor_name, dataset_path)
    removeUncommonPoints(args.detector_name, args.descriptor_name, dataset_path)
