
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
