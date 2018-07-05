import numpy as np
import cv2
import os
import parameters as para
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import glob
import itertools
import ntpath


def prep_data_gen(images_path):

    image_paths = glob.glob(images_path + "0/" + "*.jpg") + glob.glob(images_path + "0/" + "*.png") + glob.glob(
        images_path + "0/" + "*.jpeg")
    samples = len(image_paths)
    print('There are ' + str(samples) + ' samples in the folder ' + images_path + '...')
    return samples


def train_image_mask_generator(image_dir,
                               mask_dir,
                               img_rows_out,
                               img_cols_out,
                               rotation_range=20.,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               zoom_range=0.2):

    #data_gen_args = dict(rotation_range=rotation_range,
                         #width_shift_range=width_shift_range,
                         #height_shift_range=height_shift_range,
                         #zoom_range=zoom_range)
    image_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    mask_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1

    image_gen = image_datagen.flow_from_directory(
        image_dir,
        target_size=(para.img_rows, para.img_cols),
        shuffle=True,
        batch_size=para.batch_size,
        class_mode=None,
        seed=seed
    )

    mask_gen = mask_datagen.flow_from_directory(
        mask_dir,
        target_size=(img_rows_out, img_cols_out),
        shuffle=True,
        batch_size=para.batch_size,
        class_mode=None,
        color_mode="grayscale",
        seed=seed
    )

    while True:
        image_g = image_gen.next()
        mask_g = mask_gen.next()
        X = []
        Y = []
        for img, mask in zip(image_g, mask_g):
            #   transform the image by normalizing them and appending to X
            img = np.array(img).astype('uint8')
            img = img.astype(np.float32)
            img = img / 255.0
            X.append(img)
            #   transform the mask by normalizing them and appending to Y
            mask = np.array(mask).astype('uint8')
            mask = np_utils.to_categorical(mask, num_classes=para.num_classes)
            Y.append(mask)

        yield np.array(X), np.array(Y)


def test_image_mask_generator(image_dir, mask_dir, img_rows_out, img_cols_out):
    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 2

    image_gen = image_datagen.flow_from_directory(
        image_dir,
        shuffle=False,
        target_size=(para.img_rows, para.img_cols),
        batch_size=para.batch_size,
        class_mode=None,
        seed=seed
    )

    mask_gen = mask_datagen.flow_from_directory(
        mask_dir,
        shuffle=False,
        target_size=(img_rows_out, img_cols_out),
        batch_size=para.batch_size,
        class_mode=None,
        color_mode="grayscale",
        seed=seed
    )

    while True:
        image_g = image_gen.next()
        mask_g = mask_gen.next()
        X = []
        Y = []
        for img, mask in zip(image_g, mask_g):
            #   transform the image by normalizing them and appending to X
            img = np.array(img).astype('uint8')
            img = img.astype(np.float32)
            img = img / 255.0
            X.append(img)
            #   transform the mask by normalizing them and appending to Y
            mask = np.array(mask).astype('uint8')
            mask = np_utils.to_categorical(mask, num_classes=para.num_classes)
            Y.append(mask)
        yield np.array(X), np.array(Y)


def image_mask_processor(image_dir, mask_dir, img_rows, img_cols):
    image_paths = glob.glob(image_dir + "0/" + "*.jpg") + glob.glob(image_dir + "0/" + "*.png") + glob.glob(
        image_dir + "0/" + "*.jpeg")
    image_paths.sort()
    mask_paths = glob.glob(mask_dir + "0/" + "*.jpg") + glob.glob(mask_dir + "0/" + "*.png") + glob.glob(
        mask_dir + "0/" + "*.jpeg")
    mask_paths.sort()
    img = []
    msk = []
    for img_path, msk_path in zip(image_paths, mask_paths):
        image = cv2.imread(img_path)
        image = cv2.resize(image, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST)
        img.append(image)

        mask = cv2.imread(msk_path)
        mask = cv2.resize(mask, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        msk.append(mask)

    return np.array(img), np.array(msk)


def prepare_evaluation_images(image_dir, mask_dir, img_rows, img_cols, img_rows_out, img_cols_out):
    #print(image_dir)
    image_paths = glob.glob(image_dir + "0/" + "*.jpg") + glob.glob(image_dir + "0/" + "*.png") + glob.glob(
        image_dir + "0/" + "*.jpeg")
    image_paths.sort()
    #print(len(image_paths))
    mask_paths = glob.glob(mask_dir + "0/" + "*.jpg") + glob.glob(mask_dir + "0/" + "*.png") + glob.glob(
        mask_dir + "0/" + "*.jpeg")
    mask_paths.sort()
    #print(len(mask_paths))

    img = []
    msk = []
    img_names = []
    for img_path, msk_path in zip(image_paths, mask_paths):
        #   Open and resize the image to the input shape and
        #   prepare it for the classifier and the mask to the output shape
        img_names.append(ntpath.basename(img_path))
        #   Image
        image = cv2.imread(img_path)
        image = cv2.resize(image, (img_rows, img_cols), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        image = np.array(image).astype('uint8')
        image = image.astype(np.float32)
        image = image / 255.0
        img.append(image)

        #   Mask
        mask = cv2.imread(msk_path)
        mask = cv2.resize(mask, (img_rows_out, img_cols_out), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = np.array(mask).astype('uint8')
        mask = np_utils.to_categorical(mask, num_classes=para.num_classes)
        msk.append(mask)

    return np.array(img), np.array(msk), np.array(img_names)


def visualise_mask(msk, dataset=para.dataset):
    #   convert prediction to same channel as ground truth mask
    pred_mask = np.zeros((msk.shape[0], msk.shape[1], 3))
    #pred_mask[:, :, 0] = np.squeeze(msk, axis=2)
    #pred_mask[:, :, 1] = np.squeeze(msk, axis=2)
    #pred_mask[:, :, 2] = np.squeeze(msk, axis=2)

    pred_mask[:, :, 0] = msk
    pred_mask[:, :, 1] = msk
    pred_mask[:, :, 2] = msk

    # CamVid Classes = 12
    if dataset == "CamVid":
        pred_mask[np.where((pred_mask == [0, 0, 0]).all(axis=2))] = [128, 128, 128]
        pred_mask[np.where((pred_mask == [1, 1, 1]).all(axis=2))] = [128, 0, 0]
        pred_mask[np.where((pred_mask == [2, 2, 2]).all(axis=2))] = [192, 192, 128]

        pred_mask[np.where((pred_mask == [3, 3, 3]).all(axis=2))] = [128, 64, 128]
        pred_mask[np.where((pred_mask == [4, 4, 4]).all(axis=2))] = [60, 40, 222]
        pred_mask[np.where((pred_mask == [5, 5, 5]).all(axis=2))] = [128, 128, 0]

        pred_mask[np.where((pred_mask == [6, 6, 6]).all(axis=2))] = [192, 128, 128]
        pred_mask[np.where((pred_mask == [7, 7, 7]).all(axis=2))] = [64, 64, 128]
        pred_mask[np.where((pred_mask == [8, 8, 8]).all(axis=2))] = [64, 0, 128]

        pred_mask[np.where((pred_mask == [9, 9, 9]).all(axis=2))] = [64, 64, 0]
        pred_mask[np.where((pred_mask == [10, 10, 10]).all(axis=2))] = [0, 128, 192]
        pred_mask[np.where((pred_mask == [11, 11, 11]).all(axis=2))] = [0, 0, 0]

    if dataset == "flowers_multi_class":
        pred_mask[np.where((pred_mask == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
        pred_mask[np.where((pred_mask == [1, 1, 1]).all(axis=2))] = [128, 0, 0]
        pred_mask[np.where((pred_mask == [2, 2, 2]).all(axis=2))] = [192, 192, 128]

        pred_mask[np.where((pred_mask == [3, 3, 3]).all(axis=2))] = [128, 64, 128]
        pred_mask[np.where((pred_mask == [4, 4, 4]).all(axis=2))] = [60, 40, 222]
        pred_mask[np.where((pred_mask == [5, 5, 5]).all(axis=2))] = [128, 128, 0]

        pred_mask[np.where((pred_mask == [6, 6, 6]).all(axis=2))] = [192, 128, 128]
        pred_mask[np.where((pred_mask == [7, 7, 7]).all(axis=2))] = [64, 64, 128]
        pred_mask[np.where((pred_mask == [8, 8, 8]).all(axis=2))] = [64, 0, 128]

        pred_mask[np.where((pred_mask == [9, 9, 9]).all(axis=2))] = [64, 64, 0]
        pred_mask[np.where((pred_mask == [10, 10, 10]).all(axis=2))] = [0, 128, 192]
        pred_mask[np.where((pred_mask == [11, 11, 11]).all(axis=2))] = [0, 0, 128]

        pred_mask[np.where((pred_mask == [12, 12, 12]).all(axis=2))] = [64, 128, 0]

    if dataset == "flowers_fore_back":
        pred_mask[np.where((pred_mask == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
        pred_mask[np.where((pred_mask == [1, 1, 1]).all(axis=2))] = [255, 0, 255]

    if dataset == "leaf":
        pred_mask[np.where((pred_mask == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
        pred_mask[np.where((pred_mask == [1, 1, 1]).all(axis=2))] = [255, 0, 255]
        pred_mask[np.where((pred_mask == [2, 2, 2]).all(axis=2))] = [255, 255, 0]

    return np.array(pred_mask)


def plant_leaves_labels(plant="ara", images_path="plant_dataset/arabidopsis_anno/", target_path="plant_dataset/arabidopsis_anno_new/"):
    image_paths = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(
        images_path + "*.jpeg")

    if not os.path.exists(target_path):
        os.makedirs(target_path, mode=0o777)

    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    for img_path in image_paths:
        base_name = ntpath.basename(img_path)
        filename = target_path + base_name
        print(filename)
        image = cv2.imread(img_path)
        if plant == "ara":
            image[np.where((image == [0, 160, 196]).all(axis=2))] = [255, 0, 255]
        elif plant == "tobacco":
            image[np.where((image == [0, 0, 164]).all(axis=2))] = [50, 205, 50]
        cv2.imshow("image", image)
        cv2.imwrite(filename, image)
        cv2.waitKey(10)


def plant_leaves_labels_fg(color, images_path="leaf/A1annot/", target_path="leaf/A1annot_new/"):
    image_paths = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(
        images_path + "*.jpeg")

    if not os.path.exists(target_path):
        os.makedirs(target_path, mode=0o777)

    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    for img_path in image_paths:
        base_name = ntpath.basename(img_path)
        filename = target_path + base_name
        print(filename)
        image = cv2.imread(img_path)

        #image[np.where((image == [255, 255, 255]).all(axis=2))] = [2, 2, 2]
        image[np.where((image == [0, 0, 128]).all(axis=2))] = color

        cv2.imshow("image", image)
        cv2.imwrite(filename, image)
        cv2.waitKey(10)


def flowers_labels_replace(color, images_path="leaf/A1annot/", target_path="leaf/A1annot_new/"):
    image_paths = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(
        images_path + "*.jpeg")

    if not os.path.exists(target_path):
        os.makedirs(target_path, mode=0o777)

    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    for img_path in image_paths:
        base_name = ntpath.basename(img_path)
        filename = target_path + base_name
        print(filename)
        image = cv2.imread(img_path)

        image[np.where((image == [128, 0, 0]).all(axis=2))] = [0, 0, 0]
        image[np.where((image == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
        image[np.where((image == [0, 128, 128]).all(axis=2))] = [0, 0, 0]
        image[np.where((image == [0, 128, 0]).all(axis=2))] = [0, 0, 0]
        image[np.where((image == [143, 0, 0]).all(axis=2))] = [0, 0, 0]
        image[np.where((image == [191, 0, 0]).all(axis=2))] = [0, 0, 0]

        image[np.where((image == [0, 0, 128]).all(axis=2))] = color
        image[np.where((image == [159, 0, 0]).all(axis=2))] = color

        cv2.imshow("image", image)
        cv2.imwrite(filename, image)
        cv2.waitKey(10)


def flowers_copy_jpgs(images_path="flowers/gt_multi_class/trainannot/", target_path="flowers/train/",
                      src_path = "flowers/gt_fore_back/"):
    image_paths = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(
        images_path + "*.jpeg")

    if not os.path.exists(target_path):
        os.makedirs(target_path, mode=0o777)

    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    for img_path in image_paths:
        base_name = ntpath.basename(img_path).split('.')[0]
        traget_filename = target_path + str(base_name) + ".png"
        src_jpg_filename = src_path + "/" + str(base_name) + ".png"
        print(src_jpg_filename)
        image = cv2.imread(src_jpg_filename)
        #cv2.imread("jpeg", image)
        cv2.imwrite(traget_filename, image)
        cv2.waitKey(10)


def leaf_pixels_to_binary(plant="ara", images_path="leaf/testannot/", target_path="leaf/testannot_new/"):
    image_paths = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(
        images_path + "*.jpeg")

    if not os.path.exists(target_path):
        os.makedirs(target_path, mode=0o777)

    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    for img_path in image_paths:
        base_name = ntpath.basename(img_path)
        filename = target_path + base_name
        print(filename)
        image = cv2.imread(img_path)

        image[np.where((image == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
        image[np.where((image == [255, 0, 255]).all(axis=2))] = [1, 1, 1]
        image[np.where((image == [50, 205, 50]).all(axis=2))] = [2, 2, 2]
        image[np.where((image == [0, 0, 204]).all(axis=2))] = [1, 1, 1]

        cv2.imshow("image", image)
        cv2.imwrite(filename, image)
        cv2.waitKey(10)


def MeanIOU(confusion_matrix):
    I = np.diag(confusion_matrix)
    U = np.sum(confusion_matrix, axis=0) + np.sum(confusion_matrix, axis=1) - I
    IOU = I / U
    meanIOU = np.mean(IOU)
    return IOU, meanIOU


if __name__ == '__main__':
    #plant_leaves_labels(plant="tobacco", images_path="plant_dataset/tobacco_anno/",
                        #target_path="plant_dataset/tobacco_anno_new/")
    #prep_data_gen(images_path=para.mask_val_data_file_gen)
    #leaf_pixels_to_binary(plant="ara", images_path="leaf/valannot/", target_path="leaf/valannot_new/")
    import matplotlib.pyplot as plt

    image = cv2.imread("flowers_old/test/0/image_0324.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY, image)
    #print(np.array(image).shape)
    shrink_ratio = 0.75

    U, S, Vt = np.linalg.svd(np.asarray(image))
    k = int(np.array(S).shape[0] * shrink_ratio)

    cmpimg = np.matrix(U[:, :k]) * np.diag(S[:k]) * np.matrix(Vt[:k, :])
    plt.imshow(cmpimg, cmap='gray')
    title = " Image after =  %s" % k
    plt.title(title)
    plt.show()


    #print(np.unique(image, axis=2))
    #unique_pixels = np.vstack({tuple(r) for r in image.reshape(-1, 3)})

    #print(unique_pixels)

    #image[np.where((image == [159, 0, 0]).all(axis=2))] = [0, 0, 0]
    #cv2.imshow("image", image)
    #cv2.waitKey(10000)
    #load_data_in_batches(para.train_data_file_gen, para.mask_train_data_file_gen,
                                                           #para.batch_size, para.img_rows, para.img_cols,
                                                          #para.img_rows_out, para.img_cols_out)

    #train_data, train_label, train_samples = prep_data(data_file=para.train_data_file)
    #image_mask_processor(image_dir, mask_dir, img_rows, img_cols)

    #flowers_copy_jpgs(images_path="flowers/gt_multi_class/valannot/", target_path="flowers/valannot/",
                      #src_path="flowers/gt_fore_back")

    #for i in range(1, 13, 1):
        #color = [1, 1, 1]
        #flowers_labels_replace(color, images_path="flowers/gt/" + str(i-1) + "/", target_path="flowers/gt_new/" + str(i-1) + "/")

    #image_mask_processor(para.train_data_file_gen,
                                             #para.mask_train_data_file_gen,
                                             #224,
                                             #224
                                             #)