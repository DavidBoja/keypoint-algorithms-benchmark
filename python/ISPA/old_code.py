
###############################################################################################
#              VERZIJA KODA KOJI BRISE UNCOMMON POINTS SA REF SLIKE I SA SEQUENCE SLIKA
###############################################################################################

for folder in folders:
    folder_name = folder.split('/')[-1]
    print('#########################################')
    print('Working on folder {}'.format(folder_name))

    kp_file = np.load(folder + '/kp.npz')
    kp = kp_file[detector_name]
    kp = list(kp)

    des_file = np.load(folder + '/des.npz')
    nm = detector_name + '_' + descriptor_name
    des = des_file[nm]
    des = list(des)

    indexes_to_remove = []

    # iterate over sequence images, find keypoints, remove keypoints that do not appear on all the images
    for id1, kp_ in enumerate(kp):

        # remove keypoints from ref image that do not appear on sequence images
        if id1 == 0:
            print('Removing kp from image 1.ppm')

            remove = set()

            for id2, tr in enumerate(transformations[folder_name]):
                points = np.c_[ kp_[:,[0,1]] , np.ones(kp_.shape[0])]
                imaged_points = np.dot(tr, points.T)
                imaged_points_normal = imaged_points/imaged_points[2,:]

                image_size = cv2.imread(folder + '/' + str(id2+2) + '.ppm' )
                image_size = image_size.shape
#                 print(image_size)

                x_indexes_out_of_bounds = np.where((imaged_points_normal[0,:] < 0) |
                                                   (image_size[1] < imaged_points_normal[0,:]))[0]

#                 print(x_indexes_out_of_bounds)

                y_indexes_out_of_bounds = np.where((imaged_points_normal[1,:] < 0) |
                                                   (image_size[0] < imaged_points_normal[1,:]))[0]

#                 print(y_indexes_out_of_bounds)

                remove = remove.union(x_indexes_out_of_bounds)
                remove = remove.union(y_indexes_out_of_bounds)

            indexes_to_remove.append(list(remove))

        # remove keypoints from ref image that do not appear on sequence images
        else:
            print('Removing kp from image {}'.format(image_names[id1]))

            remove = set()
            image_size = cv2.imread(folder + '/1.ppm' ).shape
#             print(image_size)

            # image every keypoint on sequence image back on ref image and check if
            # it is contained in the ref image

            tr = transformations[folder_name][id1-1]

            points = np.c_[ kp_[:,[0,1]] , np.ones(kp_.shape[0])]
            imaged_points = np.dot(np.linalg.inv(tr), points.T)
            imaged_points_normal = imaged_points/imaged_points[2,:]

            x_indexes_out_of_bounds = np.where((imaged_points_normal[0,:] < 0) |
                                                   (image_size[1] < imaged_points_normal[0,:]))[0]

            y_indexes_out_of_bounds = np.where((imaged_points_normal[1,:] < 0) |
                                                   (image_size[0] < imaged_points_normal[1,:]))[0]

            remove = remove.union(x_indexes_out_of_bounds)
            remove = remove.union(y_indexes_out_of_bounds)

            indexes_to_remove.append(list(remove))


    for id2,ind in enumerate(indexes_to_remove):
        print('Removed {} keypoints from image {}'.format(len(ind), image_names[id2]))
        print('old size: {}'.format(kp[id2].shape))

        kp[id2] = np.delete(kp[id2], ind, 0)
        #np.save(kp_names[id2], kp[id2])
        print('new size: {}'.format(kp[id2].shape))

        des[id2] = np.delete(des[id2], ind, 0)
        #np.save(des_names[id2], des[id2])

    elements = dict(kp_file)
    elements[detector_name] = kp
    np.savez(folder + '/kp.npz', **elements)

    elements = dict(des_file)
    elements[nm] = des
    np.savez(folder + '/des.npz', **elements)

###############################################################################################
#                                   END
###############################################################################################

###############################################################################################
#            PATCH RETRIEVAL PRIJE ONOGA DA SVAKI KEYPOINT ZA SEBE GLEDAMO AP
###############################################################################################


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

###############################################################################################
#                                   END
###############################################################################################
