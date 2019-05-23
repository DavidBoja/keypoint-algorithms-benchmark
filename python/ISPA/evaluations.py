

import glob
import cv2
import numpy as np
import math
import re
import os
from sklearn.metrics import average_precision_score
from random import sample

from utils import getTransformations



def patchVerification(detector_name, descriptor_name, n, dataset_path, nr_of_iterations=1):
    '''


    Return: list_of_APs - list of APs per iteration
            mAP         - average AP per number of iterations
    '''

    transformations = getTransformations(dataset_path)
    folders = glob.glob(dataset_path)
    list_of_APs = []

    for i in range(nr_of_iterations):
        y = []
        s = []

        for id1, folder in enumerate(folders):

            folder_name = folder.split('/')[-1]

            # get keypoints from sequence in folder
            kp = np.load(folder + '/kp.npz')
            kp = kp[detector_name]
            kp = list(kp)
            print([kp_.shape for kp_ in kp])

            # get descriptors from sequence in folder
            des = np.load(folder + '/des.npz')
            nm = detector_name + '_' + descriptor_name
            des = des[nm]
            des = list(des)

            # get keypoints from next folder
            if id1 != (len(folders)-1):
                next_folder = folders[id1+1]
            else:
                next_folder = folders[0]

            kp_next = np.load(next_folder + '/kp.npz')
            kp_next = kp_next[detector_name]
            kp_next = list(kp_next)

            des_next = np.load(next_folder + '/des.npz')
            nm = detector_name + '_' + descriptor_name
            des_next = des_next[nm]
            des_next = list(des_next)

            # printout
            print('Working on folder {} \
                  --> next_folder {}'.format(folder_name, next_folder.split('/')[-1]))

            # check if ref image has no keypoints and skip evaluation of sequence
            if 0 == kp[0].shape[0]:
                print('Folder {} has 0 keypoints for ref image'.format(folder))
                print('SKIPPING THIS FOLDER')
                continue

            # random keypoints from ref image
            nr_of_indexes = min(n, kp[0].shape[0])
            random_keypoint_indexes = sample(range(kp[0].shape[0]), nr_of_indexes)

            # choose ref image
            x_kp = kp[0][random_keypoint_indexes,:]
            x_des = des[0][random_keypoint_indexes,:]


            # iterate over descriptors of sequence images
            for id1, dess in enumerate(des[1:]):

                # match ref image and sequence image
                bf = cv2.BFMatcher(cv2.NORM_L2)
                matches = bf.match(x_des,
                                   dess)

                # get the indexes from the matching procedure above
                x_idx = [random_keypoint_indexes[m.queryIdx] for m in matches]
                x_crtano_idx = [m.trainIdx for m in matches]

                # measure s with which we rank the AP
                s += [m.distance for m in matches]

                # image every keypoint from ref image to sequence image
                # we get Hx points as columns, 3xn matrix (third row are 1 -
                # homogen coordinates)
                tr = transformations[folder_name][id1]
                points = np.c_[ kp[0][x_idx,:][:,[0,1]] , np.ones(len(x_idx))]
                imaged_points = np.dot(tr, points.T)
                imaged_points_normal = imaged_points/imaged_points[2,:]

                # compute distance from Hx and x' (matching keypoints from sequence image)
                dist = kp[id1+1][x_crtano_idx,:][:,[0,1]].T - imaged_points_normal[[0,1],:]
                distances = np.sqrt(np.sum((dist)**2,axis=0))

                # find if x' is the closest keypoint to Hx and assign y
                for i in range(imaged_points_normal.shape[1]):
                    # dist_ is the distance from point Hx and every keypoint of image id1+1
                    diff = kp[id1+1][:,[0,1]].T - imaged_points_normal[[0,1],i].reshape(2,1)
                    dist_ = np.sqrt(np.sum((diff)**2,axis=0))
                    if (dist_ < distances[i]).any():
                        y.append(-1)
                    else:
                        y.append(1)

            # iterate over descriptors of non-sequence images
            for id2, dess in enumerate(des_next[1:]):
                bf = cv2.BFMatcher(cv2.NORM_L2)
                matches = bf.match(x_des,
                                   dess)

                s += [m.distance for m in matches]
                y += [-1 for m in matches]


        s2 = [-s_ for s_ in s]
        AP = average_precision_score(y,s2)
        list_of_APs.append(AP)

        print('| x | {} | {} | {} | {} | {} | {} | {} |'.format(detector_name,
                                                                descriptor_name,
                                                                AP,
                                                                n,
                                                                y.count(1),
                                                                y.count(-1),
                                                                n*5*len(folders)))

    mAP = sum(list_of_APs) / len(list_of_APs)

    return list_of_APs, mAP

def imageMatching():
    print('NISTA')

def patchRetrieval():
    print('NISTA2')

if __name__ == '__main__':
    import argparse

    parser_of_args = argparse.ArgumentParser(description='util functions')
    parser_of_args.add_argument('detector_name', type=str,
                                help='name of the detector')
    parser_of_args.add_argument('descriptor_name', type=str,
                                help='name of descriptor')
    parser_of_args.add_argument('n', type=int,
                                help='number of random keypoints to choose')
    parser_of_args.add_argument('dataset_path', type=str,
                                help='path to hpatches dataset')
    parser_of_args.add_argument('nr_of_iterations', type=int,
                                help='number of iterations for patch verification')

    args = parser_of_args.parse_args()

    # project_root = '/home/davidboja/PycharmProjects/FER/hpatches-benchmark/python/ISPA'
    # dataset_path = project_root + '/hpatches-sequences-release/*'

    list_of_APs, mAP = patchVerification(args.detector_name,
                                         args.descriptor_name,
                                         args.n,
                                         args.dataset_path,
                                         args.nr_of_iterations)

    # imageMatching(args['detector_name'],
    #               args['descriptor_name'],
    #               args['n'],
    #               args['dataset_path'],
    #               args['nr_of_iterations'])
    #
    # patchRetrieval(args['detector_name'],
    #                args['descriptor_name'],
    #                args['n'],
    #                args['dataset_path'],
    #                args['nr_of_iterations'])

    with open('PatchVerification.txt','a+') as file:
        file.write('det: {} | des: {} | list_of_APs: {} | mAP: {} |'.format(args.detector_name,
                                                                            args.descriptor_name,
                                                                            list_of_APs,
                                                                            mAP))
