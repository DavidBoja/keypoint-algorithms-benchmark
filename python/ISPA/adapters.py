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

        kps = [0.] * 6
        descs = [0.] * 6

        for img_result_filename in filter(lambda f: 'ppm.npz' in f, listdir(results_dir)):
            # Take index from filename as filter() doesn't keep the order.
            img_idx = int(img_result_filename[0])
            img_result_path = join(results_dir, img_result_filename)
            img_result = dict(np.load(img_result_path))

            orig_img = cv2.imread(join(folder_path, '{}.ppm'.format(img_idx)))
            processed_img = cv2.imread(join(results_dir, '{}.ppm'.format(img_idx)))

            x_scale = float(orig_img.shape[1]) / float(processed_img.shape[1])
            y_scale = float(orig_img.shape[0]) / float(processed_img.shape[0])

            img_result['kpts'][:,0] *= x_scale
            img_result['kpts'][:,1] *= y_scale

            kps[img_idx - 1] = img_result['kpts']
            descs[img_idx - 1] = img_result['descs']

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
