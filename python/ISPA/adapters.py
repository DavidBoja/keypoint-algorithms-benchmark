import numpy as np
from os import listdir
from os.path import join, isfile

from const import *

DATASET_ROOT = '../../../hpatches-sequences-release/'
KPZ_FILENAME = 'kp.npz'
DESC_FILENAME = 'des.npz'


def adapt_lfnet():
    algo_name = 'lfnet'

    for folder in listdir(DATASET_ROOT):
        print('Processing directory: {}'.format(folder))
        folder_path = join(DATASET_ROOT, folder)
        kp_path = join(folder_path, KPZ_FILENAME)
        desc_path = join(folder_path, DESC_FILENAME)
        results_dir = join(folder_path, 'out')

        kps = []
        descs = []

        for img_result_filename in filter(lambda f: 'ppm.npz' in f, listdir(results_dir)):
            img_result_path = join(results_dir, img_result_filename)
            img_result = np.load(img_result_path)

            kps.append(img_result['kpts'])
            descs.append(img_result['descs'])

        print(kps[0].shape)

        kp = dict(np.load(kp_path, allow_pickle=True)) if isfile(kp_path) else dict()
        desc = dict(np.load(desc_path, allow_pickle=True)) if isfile(desc_path) else dict()
        kp[algo_name] = np.array(kps)
        desc[ALGO_TEMPLATE.format(algo_name, algo_name)] = np.array(descs)

        np.savez(kp_path, **kp)
        np.savez(desc_path, **desc)


def adapt_superpoint():
    algo_name = 'superpoint'

    for folder in listdir(DATASET_ROOT):
        print('Processing directory: {}'.format(folder))
        folder_path = join(DATASET_ROOT, folder)
        kp_path = join(folder_path, KPZ_FILENAME)
        desc_path = join(folder_path, DESC_FILENAME)
        results_dir = join(folder_path, algo_name)

        kps = []
        descs = []

        for img_result_filename in filter(lambda f: 'ppm.npz' in f, listdir(results_dir)):
            img_result_path = join(results_dir, img_result_filename)
            img_result = np.load(img_result_path)

            kps.append(img_result['kpts'].T)
            descs.append(img_result['descs'].T)

        kp = dict(np.load(kp_path, allow_pickle=True)) if isfile(kp_path) else dict()
        desc = dict(np.load(desc_path, allow_pickle=True)) if isfile(desc_path) else dict()
        kp[algo_name] = np.array(kps)
        desc[ALGO_TEMPLATE.format(algo_name, algo_name)] = np.array(descs)

        np.savez(kp_path, **kp)
        np.savez(desc_path, **desc)


if __name__ == '__main__':
    import argparse

    parser_of_args = argparse.ArgumentParser(description='Select algorithm to adapt')

    parser_of_args.add_argument('--algorithm', type=str,
                                help='name of the algorithm')

    args = parser_of_args.parse_args()

    result = locals()['adapt_{}'.format(args.algorithm)]()
