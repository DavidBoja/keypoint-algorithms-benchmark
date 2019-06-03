

import glob
import cv2
import numpy as np
import math
import re
import os
from sklearn.metrics import average_precision_score
from random import sample

from utils import getTransformations, read_keypoints
from const import *



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
    list_of_APs_i = []
    list_of_APs_v = []

    for i in range(nr_of_iterations):
        y = []
        y_i = []
        y_v = []
        s = []
        s_i = []
        s_v = []

        for folder_id, folder in enumerate(folders):

            folder_name = folder.split('/')[-1]

            # get keypoints and descriptors from sequence in folder
            kp, des = read_keypoints(folder, detector_name, descriptor_name)

            # get keypoints from next folder
            next_folder = folders[(folder_id + 1) % len(folders)]
            kp_next, des_next = read_keypoints(next_folder,
                                               detector_name,
                                               descriptor_name)

            # printout
            print('pV: Working on folders {}--{}'.format(folder_name,
                                                         next_folder.split('/')[-1]))

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
                bf = cv2.BFMatcher(descriptor_distance[descriptor_name])
                matches = bf.match(x_des,
                                   dess)

                # get the indexes from the matching procedure above
                x_idx = [random_keypoint_indexes[m.queryIdx] for m in matches]
                x_crtano_idx = [m.trainIdx for m in matches]

                # measure s with which we rank the AP
                s += [m.distance for m in matches]
                if 'i_' in folder_name:
                    s_i += [m.distance for m in matches]
                else:
                    s_v += [m.distance for m in matches]

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
                        if 'i_' in folder_name:
                            y_i.append(-1)
                        else:
                            y_v.append(-1)
                    else:
                        y.append(1)
                        if 'i_' in folder_name:
                            y_i.append(1)
                        else:
                            y_v.append(1)

            # iterate over descriptors of non-sequence images
            for id2, dess in enumerate(des_next[1:]):
                bf = cv2.BFMatcher(descriptor_distance[descriptor_name])
                matches = bf.match(x_des,
                                   dess)

                s += [m.distance for m in matches]
                y += [-1 for m in matches]

                if 'i_' in folder_name:
                    s_i += [m.distance for m in matches]
                    y_i += [-1 for m in matches]
                else:
                    s_v += [m.distance for m in matches]
                    y_v += [-1 for m in matches]



        s2 = [-s_ for s_ in s]
        AP = average_precision_score(y,s2)
        list_of_APs.append(AP)

        s_i = [-s_ for s_ in s_i]
        AP_i = average_precision_score(y_i,s_i)
        list_of_APs_i.append(AP_i)

        s_v = [-s_ for s_ in s_v]
        AP_v = average_precision_score(y_v,s_v)
        list_of_APs_v.append(AP_v)


    return list_of_APs, list_of_APs_i, list_of_APs_v


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
    list_of_mAPs_i = []
    list_of_mAPs_v = []

    for i in range(nr_of_iterations):

        list_of_APs = []
        list_of_APs_i = []
        list_of_APs_v = []

        for folder in folders:

            folder_name = folder.split('/')[-1]
            print('iM: Working on folder {}'.format(folder_name))

            # get keypoints and descriptors from sequence
            kp, des = read_keypoints(folder, detector_name, descriptor_name)

            # if an image has no keypoints skip that evaluating sequence
            if 0 == kp[0].shape[0]:
                print('Folder {} has 0 keypoints for ref image'.format(folder))
                print('SKIPPING THIS FOLDER')
                continue

            # random keypoints from ref image
            nr_of_indexes = min(n,kp[0].shape[0])
            random_keypoint_indexes = sample(range(kp[0].shape[0]), nr_of_indexes)
            #print('NR OF INDEXES: {}'.format(nr_of_indexes))


            # define ref image
            x_kp = kp[0][random_keypoint_indexes,:]
            x_des = des[0][random_keypoint_indexes,:]


            for id1, dess in enumerate(des[1:]):
                y = []
                s = []

                # match ref image and sequence image
                bf = cv2.BFMatcher(descriptor_distance[descriptor_name])
                matches = bf.match(x_des,
                                   dess)

                # get indexes of matches
                x_idx = [random_keypoint_indexes[m.queryIdx] for m in matches]
                x_crtano_idx = [m.trainIdx for m in matches]
                #print(x_crtano_idx)
                #print(kp[id1+1].shape)

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

                # compute AP for pair of images if there is at least one positive match
                # if there is no positive match the AP will be nan so we cant compute mAP
                if 1 in y:
                    s2 = [-s_ for s_ in s]
                    AP = average_precision_score(y,s2)
                    list_of_APs.append(AP)
                    if 'i_' in folder_name:
                        list_of_APs_i.append(AP)
                    else:
                        list_of_APs_v.append(AP)

        if list_of_APs:
            list_of_mAPs.append(sum(list_of_APs) / len(list_of_APs))
            list_of_mAPs_i.append(sum(list_of_APs_i) / len(list_of_APs_i))
            list_of_mAPs_v.append(sum(list_of_APs_v) / len(list_of_APs_v))
        else:
            list_of_mAPs.append(0)
            list_of_all_APs.append(list_of_APs)

    return list_of_mAPs, list_of_mAPs_i, list_of_mAPs_v


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
    list_of_mAPs_i = []
    list_of_mAPs_v = []

    for i in range(nr_of_iterations):

        list_of_APs = []
        list_of_APs_i = []
        list_of_APs_v = []

        for folder_id, folder in enumerate(folders):

            folder_name = folder.split('/')[-1]

            # get keypoints and descriptors from sequence in folder
            kp, des = read_keypoints(folder, detector_name, descriptor_name)

            # get keypoints from next folder
            next_folder = folders[(folder_id + 1) % len(folders)]
            kp_next, des_next = read_keypoints(next_folder,
                                               detector_name,
                                               descriptor_name)

            print('pR: Working on folders {}--{}'.format(folder_name,
                                                         next_folder.split('/')[-1]))

            # check if an image has no keypoints and skip that evaluating sequence
            if 0 == kp[0].shape[0]:
                print('Folder {} has 0 keypoints for ref image'.format(folder))
                print('SKIPPING THIS FOLDER')
                continue

            # random keypoints from ref image
            nr_of_indexes = min(n,kp[0].shape[0])
            random_keypoint_indexes = sample(range(kp[0].shape[0]), nr_of_indexes)
            #print('NR OF INDEXES: {}'.format(nr_of_indexes))

            # create dict which saves y and s for every keypoint separately
            y = {}
            s = {}
            for i in range(len(random_keypoint_indexes)):
                y[i] = []
                s[i] = []

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


                # for every column in Hx, find its closest kp on the sequence image
                for i in range(imaged_points_normal.shape[1]):
                    # computing distance between all kp and finding the minimal
                    dist = kp[id1+1][:,[0,1]].T - imaged_points_normal[[0,1],i].reshape(2,1)
                    distances = np.sqrt(np.sum((dist)**2,axis=0))
                    index_of_closest_kp = np.argmin(distances)
                    #closest_keypoint = kp[id1+1][index_of_closest_kp,:]

                    y[i].append(1)
                    # find distance between descriptors and save as measure s
                    index_in_orig_kp0 = random_keypoint_indexes[i]
                    descriptors_distance = cv2.norm(dess[index_of_closest_kp,:],
                                                    des[0][index_in_orig_kp0,:],
                                                    descriptor_distance[descriptor_name])
                    s[i].append(descriptors_distance)


            # images out of the sequence
            for dess in des_next[1:]:

                # obtain random points from second image
                how_many = min(n,dess.shape[0])
                random_keypoint_indexes_right_img = sample(range(dess.shape[0]), how_many)

                # match points
                bf = cv2.BFMatcher(descriptor_distance[descriptor_name])
                matches = bf.match(x_des,
                                   dess[random_keypoint_indexes_right_img,:])

                for m in matches:
                    s[m.queryIdx].append(m.distance)
                    y[m.queryIdx].append(-1)


            # after iterating over sequence, compute AP
            for i in range(len(y.keys())):
                s2 = [-s_ for s_ in s[i]]
                AP = average_precision_score(y[i],s2)
                list_of_APs.append(AP)

                if 'i_' in folder_name:
                    list_of_APs_i.append(AP)
                else:
                    list_of_APs_v.append(AP)

        list_of_mAPs.append(sum(list_of_APs) / len(list_of_APs))
        list_of_mAPs_i.append(sum(list_of_APs_i) / len(list_of_APs_i))
        list_of_mAPs_v.append(sum(list_of_APs_v) / len(list_of_APs_v))

    return list_of_mAPs, list_of_mAPs_i, list_of_mAPs_v


def write_result(l, f):
    s = '['
    for e in l:
        s += str(e) + ', '
    s += ']'
    f.write(s)


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

    list_of_APs_pV, list_of_APs_i_pV, list_of_APs_v_pV = patchVerification(args.detector_name,
                                                         args.descriptor_name,
                                                         args.n,
                                                         args.dataset_path,
                                                         args.nr_of_iterations)

    list_of_mAPs_iM, list_of_mAPs_i_iM, list_of_mAPs_v_iM = imageMatching(args.detector_name,
                                                                          args.descriptor_name,
                                                                          args.n,
                                                                          args.dataset_path,
                                                                          args.nr_of_iterations)

    list_of_mAPs_pR, list_of_mAPs_i_pR, list_of_mAPs_v_pR = patchRetrieval(args.detector_name,
                                                                           args.descriptor_name,
                                                                           args.n,
                                                                           args.dataset_path,
                                                                           args.nr_of_iterations)

    print(list_of_APs_pV)
    print(list_of_APs_i_pV)
    print(list_of_APs_v_pV)

    print(list_of_mAPs_iM)
    print(list_of_mAPs_i_iM)
    print(list_of_mAPs_v_iM)

    print(list_of_mAPs_pR)
    print(list_of_mAPs_i_pR)
    print(list_of_mAPs_v_pR)

    with open(os.path.join(RESULTS_DIR, 
        ALGO_TEMPLATE.format(args.detector_name, args.descriptor_name)), 'w+') as f:
        f.write('Patch verification:\n')
        write_result(list_of_APs_pV, f)
        write_result(list_of_APs_i_pV, f)
        write_result(list_of_APs_v_pV, f)
        f.write('\nImage matching:\n')
        write_result(list_of_mAPs_iM, f)
        write_result(list_of_mAPs_i_iM, f)
        write_result(list_of_mAPs_v_iM, f)
        f.write('\nPatch retrieval:\n')
        write_result(list_of_mAPs_pR, f)
        write_result(list_of_mAPs_i_pR, f)
        write_result(list_of_mAPs_v_pR, f)

    # TODO: izbrojati broj keypointova za svaki algo i vidjeti postoji li korelacija