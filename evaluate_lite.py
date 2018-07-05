import parameters as para
from Models import resnet, resnet50_like, fcn, fcn_lite, resnet_lite, resnet50_like_lite
import dataset_utils
import cv2, os
from custom_metrics import *


if __name__ == '__main__':
    alphas = [0.25, 0.5, 0.75, 1]
    for alpha in alphas:
        #   Call model
        #   ==========
        output_rows = 0
        output_cols = 0

        if para.model_type == "fcn_lite":
            print('Initialising FCN Lite ' + str(alpha) + '...')
            model, output_rows, output_cols = fcn_lite.fcn_8(alpha=alpha)

        if para.model_type == "resnet_18_lite":
            print('Initialising ResNet 18 Lite ' + str(alpha) + '...')
            model, output_rows, output_cols = resnet_lite.ResNet18(alpha=alpha)

        if para.model_type == "resnet_34_lite":
            print('Initialising ResNet 34 Lite ' + str(alpha) + '...')
            model, output_rows, output_cols = resnet_lite.ResNet34(alpha=alpha)

        if para.model_type == "resnet50_like_lite":
            print('Initialising ResNet-50-Like Lite ' + str(alpha) + '...')
            model, output_rows, output_cols = resnet50_like_lite.ResNet50_like(alpha=alpha)


        #   Compile the model using sgd optimizer
        #   =====================================
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[dice_coef, precision, recall, 'accuracy'])
        print(model.summary())
        print(para.misc_dir_eval)
        # Load the model weights
        model.load_weights(para.misc_dir_eval + str(alpha) + "/weights.h5")

        #   Get the evaluation images and labels
        test_images, test_masks, test_image_names = dataset_utils.prepare_evaluation_images(image_dir=para.val_data_file_gen, mask_dir=para.mask_val_data_file_gen,
                                                img_rows=para.img_rows, img_cols=para.img_cols, img_rows_out=output_rows,
                                                img_cols_out=output_cols)
        #   Evaluate and print the score
        print("Evaluating test data and computing pixel accuracy...")
        scores = model.evaluate(test_images, test_masks, verbose=1, batch_size=para.batch_size)
        print("%s: %.2f%%" % ("Dice Score: ", scores[1]*100))
        print("%s: %.2f%%" % ("Precision: ", scores[2] * 100))
        print("%s: %.2f%%" % ("Recall: ", scores[3] * 100))
        print("%s: %.2f%%" % ("Pixel Accuracy: ", scores[4] * 100))

        #   Perform predictions, display results and save results
        pred_mask = model.predict(test_images, batch_size=para.batch_size)
        mean_iou = MeanIOU(test_masks, pred_mask)
        print("%s: %.2f%%" % ("Mean IoU: ", mean_iou * 100))

        '''Visualise the predictions and save results'''
        image_path = para.misc_dir + str(alpha) + "/results"

        if not os.path.exists(image_path):
            os.makedirs(image_path, mode=0o777)

        prediction_k = []
        f = open(image_path + "/" + 'mean_iou.csv', 'w')
        f.write("Dice Score: , " + str(scores[1]*100) + "\n")
        f.write("Precision: , " + str(scores[2] * 100) + "\n")
        f.write("Recall: , " + str(scores[3] * 100) + "\n")
        f.write("Pixel Accuracy: , " + str(scores[4] * 100) + "\n")
        f.write("Mean IoU: , " + str(mean_iou * 100) + "\n")

        for p_mask, gt_mask, t_image, img_names in zip(pred_mask, test_masks, test_images, test_image_names):
            f.write(img_names + "," + str(MeanIOU(gt_mask, p_mask) * 100) + "\n")

            prediction = np.argmax(p_mask, axis=1).reshape((para.img_cols, para.img_rows))
            ground_truth = np.argmax(gt_mask, axis=1).reshape((para.img_cols, para.img_rows))

            pred = dataset_utils.visualise_mask(prediction)
            g_truth = dataset_utils.visualise_mask(ground_truth)

            pred = cv2.copyMakeBorder(pred, 40, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            pred = cv2.putText(pred, "Pred. Mask", (20, 20), cv2.FONT_ITALIC, 0.6, 255)

            g_truth = cv2.copyMakeBorder(g_truth, 40, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            g_truth = cv2.putText(g_truth, "GT Mask", (20, 20), cv2.FONT_ITALIC, 0.6, 255)

            t_image_disp = cv2.copyMakeBorder(t_image, 40, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            t_image_disp = cv2.putText(t_image_disp, "Image", (20, 20), cv2.FONT_ITALIC, 0.6, 255)

            img_gt_pred = np.concatenate((t_image_disp, g_truth, pred), axis=1)
            #cv2.imshow("Results", img_gt_pred)

            t_image = t_image * 255.0
            t_image = t_image.astype(np.ubyte)
            t_image = cv2.copyMakeBorder(t_image, 40, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            t_image = cv2.putText(t_image, "Image", (20, 20), cv2.FONT_ITALIC, 0.6, 255)
            img_gt_pred_save = np.concatenate((t_image, g_truth, pred), axis=1)
            cv2.imwrite(image_path + "/" + img_names, img_gt_pred_save)

            #cv2.waitKey(3000)

        f.close()