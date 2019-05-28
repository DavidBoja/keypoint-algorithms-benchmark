
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import math
import glob
import re
import os
from matplotlib.patches import ConnectionPatch
from pprint import pprint
from sklearn.metrics import average_precision_score
from random import sample

from const import *


def visualizeKp(detector_name, folder_path):
    '''
    Visualize keypoints of a detector on a sequence in a given folder.
    '''

    sequence_images = glob.glob(folder_path + '/*.ppm')
    sequence_images = sorted(sequence_images)
    image_names = [im.split('/')[-1] for im in sequence_images]
    sequence_images = [cv2.imread(im) for im in sequence_images]

    kp = np.load(folder_path + '/kp.npz')
    kp = kp[detector_name]
    kp = list(kp)

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20,20))
    for id1, axx in enumerate(ax.ravel()):

        gray = cv2.cvtColor(sequence_images[id1] ,cv2.COLOR_RGB2GRAY)
    #     gray_with_kp = cv2.drawKeypoints(gray,
    #                                      kp[id1],
    #                                      cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    #     axx.imshow(gray_with_kp)

        axx.imshow(gray)
        axx.scatter(kp[id1][:,0],kp[id1][:,1], s=1, color='red')
        axx.set_title('{}'.format(image_names[id1]))
    plt.show()


def topNrMatches(detector_name, descriptor_name, folder_path, nr_matches):
    '''
    Draw grid of 5x2 images where the left one is always the ref image (1.ppm)
    and the right ones are 2-6.ppm. Draw lines between matching keypoints.
    '''

    # load images
    sequence_images = glob.glob(folder_path + '/*.ppm')
    sequence_images = sorted(sequence_images)
    image_names = [im.split('/')[-1] for im in sequence_images]
    sequence_images = [cv2.imread(im) for im in sequence_images]

    # load detectors
    kp = np.load(folder_path + '/kp.npz')
    kp = kp[detector_name]
    kp = list(kp)

    # load descriptors
    des = np.load(folder_path + '/des.npz')
    nm = detector_name + '_' + descriptor_name
    des = des[nm]
    des = list(des)

    # compute matches
    matches_with_ref = []

    for dess in des[1:]:
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.match(des[0], dess)
        matches_with_ref.append(sorted(matches, key = lambda x:x.distance))

    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(30,30))
    for i, axx in enumerate(ax.ravel()):
        if (i % 2) == 1:
            continue

        points_on_first_img = []
        points_on_second_img = []

        axx.imshow(sequence_images[0])
        axx.set_title('{}'.format(image_names[0]))
        ax.ravel()[i+1].imshow(sequence_images[int(i/2)+1])
        ax.ravel()[i+1].set_title('{}'.format(image_names[int(i/2)+1]))

        for m in matches_with_ref[int(i/2)][:nr_matches]:
            points_on_first_img.append(kp[0][m.queryIdx,[0,1]])
            points_on_second_img.append(kp[int(i/2)+1][m.trainIdx,[0,1]])

        axx.scatter([i[0] for i in points_on_first_img],
                    [i[1] for i in points_on_first_img],
                    s=20,
                    c='red')

        ax.ravel()[i+1].scatter([i[0] for i in points_on_second_img],
                                [i[1] for i in points_on_second_img],
                                s=20,
                                c='red')

        for j in range(len(points_on_first_img)):
            con = ConnectionPatch(xyA=(points_on_second_img[j][0],points_on_second_img[j][1]),
                                  xyB=(points_on_first_img[j][0],points_on_first_img[j][1]),
                                  coordsA="data", coordsB="data",
                                  axesA=ax.ravel()[i+1], axesB=axx, color="red")

            ax.ravel()[i+1].add_artist(con)


    plt.show()


def taskEvaluation(dict_of_APs):
    '''
    Returns a barchart with the keys of the dict_of_APs as names of algorithms
    and height as mAP from dict_of_APs values. Aditionally plot dots for min
    and max values on barchart.
    '''

    names = []
    scores = []
    err = []

    for alg,results in dict_of_APs.items():
        names.append(alg)
        mean_ = np.mean(results)
        scores.append(float('{0:.2f}'.format(mean_)))
        err.append([float('{0:.2f}'.format(mean_ - np.min(results))), 
                    float('{0:.2f}'.format(np.max(results) - mean_))])

    pos = np.arange(len(dict_of_APs))

    fig, ax1 = plt.subplots(ncols=3, figsize=(10,10))
    fig.subplots_adjust(left=0.1, right=0.9)


    for i in range(3):
        ax1[i].barh(
                pos, 
                scores,
                align='center',
                height=0.5,
                tick_label=names if i == 0 else ['' for _ in range(len(dict_of_APs))],
                color=seaborn.color_palette("Set2"),
                xerr=np.array(err).T,
                capsize=5.)
        plt.xlim([0, 1])
        #ax2 = ax1[i].twinx()
        #ax2.set_yticks(pos)
        #ax2.set_ylim(ax1[i].get_ylim())
        #ax2.set_yticklabels(scores)

        ax1[i].set_title(TASKS[i])
    plt.savefig('graph.pdf', format='pdf')


if __name__ == '__main__':
    import argparse

    # parser_of_args = argparse.ArgumentParser(description='Visualize keypoints and graphs')
    # parser_of_args.add_argument('detector_name', type=str,
    #                             help='Name of the detector')
    # parser_of_args.add_argument('descriptor_name', type=str,
    #                             help='Name of the descriptor')
    # parser_of_args.add_argument('folder_path', type=str,
    #                             help='Path of folder')
    # parser_of_args.add_argument('nr_matches', type=int,
    #                             help='Number of matches to visualize')
    #
    # args = parser_of_args.parse_args()

    # visualizeKp(args.detector_name, args.folder_path)
    # topNrMatches(args.detector_name, args.descriptor_name, args.folder_path, args.nr_matches) 

    dict_of_APs = { 
                    SIFT: np.random.rand(5),
                    SURF : np.random.rand(5),
                    BRISK : np.random.rand(5),
                    BRIEF : np.random.rand(5),
                    ORB : np.random.rand(5),
                    ALGO_TEMPLATE.format(SIFT, SURF) : np.random.rand(5),
                    ALGO_TEMPLATE.format(BRISK, BRIEF) : np.random.rand(5),
                    SHI_TOMASI : np.random.rand(5)
                  }


    taskEvaluation(dict_of_APs)
