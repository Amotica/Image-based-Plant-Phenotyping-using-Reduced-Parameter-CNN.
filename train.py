from keras import optimizers
from keras.callbacks import *
import matplotlib.pyplot as plt
import parameters as para
from Models import resnet, resnet50_like, fcn
import numpy as np
from utils.memory_usage import memory_usage
import dataset_utils
from keras.utils import np_utils
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    #   Call model
    #   ==========
    output_rows = 0
    output_cols = 0

    if para.model_type == "fcn":
        print('Initialising FCN...')
        model, output_rows, output_cols = fcn.fcn_8()

    if para.model_type == "resnet50_like":
        print('Initialising RestNet 50 like...')
        model, output_rows, output_cols = resnet50_like.ResNet50_like()

    if para.model_type == "resnet_18":
        print('Initialising RestNet 18 ...')
        model, output_rows, output_cols = resnet.ResNet18()

    if para.model_type == "resnet_34":
        print('Initialising RestNet 34 ...')
        model, output_rows, output_cols = resnet.ResNet34()

    #   Save the model after every epoch. NB: Save the best only
    #   ========================================================
    # Create folder if not exists
    models_folder = para.home_dir + 'misc/' + para.dataset + '/' + para.model_type
    if not os.path.exists(models_folder):
        os.makedirs(models_folder, mode=0o777)

    models_log_folder = para.home_dir + 'misc/' + para.dataset + '/' + para.model_type + '/tensorboard_log'
    if not os.path.exists(models_log_folder):
        os.makedirs(models_log_folder, mode=0o777)

    #   Set up callbacks to be assigned to the fitting stage
    check_pt_file = models_folder + '/weights.h5'
    model_checkpoint = ModelCheckpoint(check_pt_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10, verbose=1, mode='auto',
                                      epsilon=0.0001, cooldown=0, min_lr=0)
    tensorboard_log = TensorBoard(log_dir=models_log_folder + '/', histogram_freq=0,
                                  write_graph=True, write_images=True)

    #   Load the training and validation dataset
    #   ========================================
    train_samples = dataset_utils.prep_data_gen(images_path=para.train_data_file_gen)
    steps_per_epoch = int(train_samples / para.batch_size)

    val_samples = dataset_utils.prep_data_gen(images_path=para.val_data_file_gen)
    validation_steps = int(val_samples / para.batch_size)

    #   Compile the model using sgd optimizer
    #   =====================================
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    print(model.summary())

    #   print the approximate memory requirement by the model
    #   =====================================================
    gigabytes = memory_usage(model, batch_size=para.batch_size)
    print('Model will need ', gigabytes, 'GB of Memory')

    if para.data_gen:
        #   Load the training dataset
        train_batches = dataset_utils.train_image_mask_generator(para.train_data_file_gen,
                                                                 para.mask_train_data_file_gen,
                                                                 output_rows,
                                                                 output_cols
                                                                 )
        #   Load the validation dataset
        test_batches = dataset_utils.test_image_mask_generator(para.test_data_file_gen,
                                                               para.mask_test_data_file_gen,
                                                               output_rows,
                                                               output_cols
                                                               )

        #   fit the model on the validation and training data with data augmentation
        #   ========================================================================
        history = model.fit_generator(
            generator=train_batches,
            steps_per_epoch=steps_per_epoch,
            epochs=para.num_epoch,
            validation_data=test_batches,
            validation_steps=validation_steps,
            class_weight=para.class_weighting,
            callbacks=[model_checkpoint, tensorboard_log, reduce_lr]
        )
    else:
        #   Load the training dataset
        #   =========================
        train_images, train_labels = dataset_utils.image_mask_processor(para.train_data_file_gen,
                                                                        para.mask_train_data_file_gen,
                                                                        para.img_rows, para.img_cols
                                                                        )
        #print(len(train_images))
        #print(len(train_labels))
        train_labels = np_utils.to_categorical(train_labels, para.num_classes)
        train_labels = np.reshape(train_labels, (len(train_images), para.img_rows * para.img_cols, para.num_classes))
        steps_per_epoch = len(train_images) / para.batch_size
        train_images = train_images.astype(np.float32)
        train_images = train_images / 255.0


        #   Load the test dataset
        #   =====================
        test_images, test_labels = dataset_utils.image_mask_processor(para.test_data_file_gen,
                                                                      para.mask_test_data_file_gen,
                                                                      para.img_rows, para.img_cols
                                                                      )
        #print(len(test_images))
        #print(len(test_labels))
        test_labels = np_utils.to_categorical(test_labels, para.num_classes)
        test_labels = np.reshape(test_labels, (len(test_images), para.img_rows * para.img_cols, para.num_classes))
        test_images = test_images.astype(np.float32)
        test_images = test_images / 255.0


        history = model.fit(train_images,
                            train_labels,
                            callbacks=[model_checkpoint, tensorboard_log, reduce_lr],
                            batch_size=para.batch_size,
                            epochs=para.num_epoch,
                            verbose=1,
                            class_weight=para.class_weighting,
                            validation_data=(test_images, test_labels),
                            shuffle=True
                            )

    #   summarize history for accuracy
    #   ==============================
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='upper left')
    plt.savefig(para.home_dir + 'misc/' + para.dataset + '/' + para.model_type + '/accuracy.png')
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    #   summarize history for accuracy
    #   ==============================
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper left')
    plt.savefig(para.home_dir + 'misc/' + para.dataset + '/' + para.model_type + '/loss.png')
    plt.show()
