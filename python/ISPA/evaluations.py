

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
    Task 1: Patch Verification
    + save results to patchVerification.txt

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

        # print result for every iteration
        # print('| x | {} | {} | {} | {} | {} | {} | {} |'.format(detector_name,
        #                                                         descriptor_name,
        #                                                         AP,
        #                                                         n,
        #                                                         y.count(1),
        #                                                         y.count(-1),
        #                                                         n*5*len(folders)))

    #mAP = sum(list_of_APs) / len(list_of_APs)

    with open('patchVerification.txt','a+') as file:
        file.write('det: {} | des: {} | list_of_APs: {} |'.format(detector_name,
                                                                            descriptor_name,
                                                                            list_of_APs
                                                                            ))

    return list_of_APs


def imageMatching(detector_name, descriptor_name, n, dataset_path, nr_of_iterations=1):
    '''
    Task 2: Image Matching.
    + save results to imageMatching.txt

    Return: list_of_APs: list of average precision for every iteration
            mAP: mean of list_of_APs
    '''

    transformations = getTransformations(dataset_path)
    folders = glob.glob(dataset_path)
    list_of_mAPs = []
    list_of_all_APs = []

    for i in range(nr_of_iterations):

        list_of_APs = []

        for id1, folder in enumerate(folders):

            folder_name = folder.split('/')[-1]
            print('Working on folder {}'.format(folder_name,))

            # get keypoints from sequence
            kp = np.load(folder + '/kp.npz')
            kp = kp[detector_name]
            kp = list(kp)

            # get descriptors from sequence
            des = np.load(folder + '/des.npz')
            nm = detector_name + '_' + descriptor_name
            des = des[nm]
            des = list(des)


            # if an image has no keypoints skip that evaluating sequence
            if 0 == kp[0].shape[0]:
                print('Folder {} has 0 keypoints for ref image'.format(folder))
                print('SKIPPING THIS FOLDER')
                continue

            # random keypoints from ref image
            nr_of_indexes = min(n,kp[0].shape[0])
            random_keypoint_indexes = sample(range(kp[0].shape[0]), nr_of_indexes)
            print('NR OF INDEXES: {}'.format(nr_of_indexes))


            # define ref image
            x_kp = kp[0][random_keypoint_indexes,:]
            x_des = des[0][random_keypoint_indexes,:]


            for id1, dess in enumerate(des[1:]):

                y = []
                s = []

                # match ref image and sequence image
                bf = cv2.BFMatcher(cv2.NORM_L2)
                matches = bf.match(x_des,
                                   dess)

                # get indexes of matches
                x_idx = [random_keypoint_indexes[m.queryIdx] for m in matches]
                x_crtano_idx = [m.trainIdx for m in matches]

                # measure s used for AP
                s += [m.distance for m in matches]

                # image every keypoint on ref image on
                # Hx are saved in columns, 3xn matrix (third row are all ones)
                tr = transformations[folder_name][id1]
                points = np.c_[ kp[0][x_idx,:][:,[0,1]] , np.ones(len(x_idx))]
                imaged_points = np.dot(tr, points.T)
                imaged_points_normal = imaged_points/imaged_points[2,:]

                # compute distance from Hx and x'
                dist = kp[id1+1][x_crtano_idx,:][:,[0,1]].T - imaged_points_normal[[0,1],:]
                distances = np.sqrt(np.sum((dist)**2,axis=0))

                # find if x' is the closest keypoint to Hx and assign y
                for i in range(imaged_points_normal.shape[1]):
                    # dist_ is the distance from point Hx and every keypoint of image id1+1
                    diff = kp[id1+1][:,[0,1]].T - imaged_points_normal[[0,1],i].reshape(2,1)
                    dist_ = np.sqrt(np.sum((diff)**2,axis=0))
                    # if there is any keypoint closer to Hx than our matching one, y=-1
                    if (dist_ < distances[i]).any():
                        y.append(-1)
                    else:
                        y.append(1)

                # compute AP for pair of images
                s2 = [-s_ for s_ in s]
                AP = average_precision_score(y,s2)
                list_of_APs.append(AP)
                # print result for every pair of images
                # print('| x | {} | {} | {} | {} | {} | {} | {} |'.format(detector_name,
                #                                                         descriptor_name,
                #                                                         AP,
                #                                                         n,
                #                                                         y.count(1),
                #                                                         y.count(-1),
                #                                                         n*5*len(folders)))




        list_of_mAPs.append(sum(list_of_APs) / len(list_of_APs))
        list_of_all_APs.append(list_of_APs)

    with open('imageMatching.txt','a+') as file:
        file.write('det: {} | des: {} | list_of_all_APs: {} | list_of_mAPs: {} |'.format(detector_name,
                                                                                           descriptor_name,
                                                                                           list_of_all_APs,
                                                                                           list_of_mAPs))

    return list_of_all_APs, list_of_mAPs


def patchRetrieval(detector_name, descriptor_name, n, dataset_path, nr_of_iterations=1):
    '''
    Task 3: Patch Retrieval.
    + save results to patchRetrieval.txt

    Return: list_of_APs: list of average precision for every iteration
            mAP: mean of list_of_APs
    '''

    transformations = getTransformations(dataset_path)
    folders = glob.glob(dataset_path)
    list_of_mAPs = []
    list_of_all_APs = []

    for i in range(nr_of_iterations):

        list_of_APs = []

        for id1, folder in enumerate(folders):

            y = []
            s = []

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

            print('Working on folder {} --> next_folder {}'.format(folder_name,next_folder.split('/')[-1]))

            kp_next = np.load(next_folder + '/kp.npz')
            kp_next = kp_next[detector_name]
            kp_next = list(kp_next)

            des_next = np.load(next_folder + '/des.npz')
            nm = detector_name + '_' + descriptor_name
            des_next = des_next[nm]
            des_next = list(des_next)

            # check if an image has no keypoints and skip that evaluating sequence
            if 0 == kp[0].shape[0]:
                print('Folder {} has 0 keypoints for ref image'.format(folder))
                print('SKIPPING THIS FOLDER')
                continue

            # random keypoints from ref image
            nr_of_indexes = min(n,kp[0].shape[0])
            random_keypoint_indexes = sample(range(kp[0].shape[0]), nr_of_indexes)
            print('NR OF INDEXES: {}'.format(nr_of_indexes))

            # choose ref image
            x_kp = kp[0][random_keypoint_indexes,:]
            x_des = des[0][random_keypoint_indexes,:]


            for id1, dess in enumerate(des[1:]):

                # image every keypoint from ref image onto sequence image
                # Hx are saved in columns, 3xn matrix (third row are all ones)
                tr = transformations[folder_name][id1]
                points = np.c_[ x_kp[:,[0,1]] , np.ones(x_kp.shape[0])]
                imaged_points = np.dot(tr, points.T)
                imaged_points_normal = imaged_points/imaged_points[2,:]


                # for every column in Hx, find its closes kp on the sequence image
                for i in range(imaged_points_normal.shape[1]):
                    # computing distance between all kp and finding the minimal
                    dist = kp[id1+1][:,[0,1]].T - imaged_points_normal[[0,1],i].reshape(2,1)
                    distances = np.sqrt(np.sum((dist)**2,axis=0))
                    index_of_closest_kp = np.argmin(distances)
                    #closest_keypoint = kp[id1+1][index_of_closest_kp,:]

                    y.append(1)
                    # TODO: ADD cv2.HAMMING distance for binary detectors
                    index_in_orig_kp0 = random_keypoint_indexes[i]
                    descriptors_distance = dess[index_of_closest_kp,:] - des[0][index_in_orig_kp0,:]
                    s.append(np.sqrt(np.sum((descriptors_distance)**2,axis=0)))



            for id2, dess in enumerate(des_next[1:]):
                bf = cv2.BFMatcher(cv2.NORM_L2)
                matches = bf.match(x_des,
                                   dess)

                s += [m.distance for m in matches]
                y += [-1 for m in matches]


            # after iterating over sequence, compute AP
            s2 = [-s_ for s_ in s]
            AP = average_precision_score(y,s2)
            list_of_APs.append(AP)
            list_of_all_APs.append(AP)

            # print result for every sequence
            # print('| x | {} | {} | {} | {} | {} | {} | {} |'.format(detector_name,
            #                                                         descriptor_name,
            #                                                         AP,
            #                                                         n,
            #                                                         y.count(1),
            #                                                         y.count(-1),
            #                                                         n*5*len(folders)))

        list_of_mAPs.append(sum(list_of_APs) / len(list_of_APs))

    with open('patchRetrieval.txt','a+') as file:
        file.write('det: {} | des: {} | list_of_APs: {} | list_of_mAPs: {} |'.format(detector_name,
                                                                                     descriptor_name,
                                                                                     list_of_all_APs,
                                                                                     list_of_mAPs))

    return list_of_all_APs, list_of_mAPs


if __name__ == '__main__':
    import argparse

    parser_of_args = argparse.ArgumentParser(description='Evaluate your algorithm on 3 tasks')
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

    list_of_APs_pV = patchVerification(args.detector_name,
                                       args.descriptor_name,
                                       args.n,
                                       args.dataset_path,
                                       args.nr_of_iterations)

    list_of_all_APs_iM, list_of_mAPs_iM = imageMatching(args.detector_name,
                                                        args.descriptor_name,
                                                        args.n,
                                                        args.dataset_path,
                                                        args.nr_of_iterations)

    list_of_all_APs_pR, list_of_mAPs_pR = patchRetrieval(args.detector_name,
                                                         args.descriptor_name,
                                                         args.n,
                                                         args.dataset_path,
                                                         args.nr_of_iterations)
