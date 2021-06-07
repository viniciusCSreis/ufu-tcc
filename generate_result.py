import os

import cv2
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

from TCC.model_util import load_all_imgs

if __name__ == '__main__':
    v2_result_path = '../imagens_cra/result/v2'
    v2_folders = os.listdir(v2_result_path)
    v2_folders = list(filter(lambda f: str(f).startswith("unet_multiclass"), v2_folders))

    x_dir = "../imagens_cra/train/cra"

    metric_name = "mean_iou_threshold"
    metric = tf.keras.metrics.MeanIoU(num_classes=2)
    threshold = 0.5


    def mean_iou_threshold(y_true, y_pred):
        y_pred = y_pred.numpy()
        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = 0
        metric.reset_states()
        metric.update_state(y_true, y_pred)
        return metric.result().numpy()


    for folder in v2_folders:

        y_dir = "../imagens_cra/validation_interna/cra"
        if folder.__contains__("external"):
            y_dir = "../imagens_cra/validation/cra"
        if folder.__contains__("multiclass"):
            y_dir = "../imagens_cra/validation_interna_externa_v2/cra"

        new_img_path = os.path.join(v2_result_path, folder, "result-images-train-size")
        print("new_img_path:", new_img_path)
        try:
            os.mkdir(new_img_path)
        except OSError as error:
            pass

        img_size = (64, 64)
        if folder.__contains__("(128, 128)"):
            img_size = (128, 128)
        if folder.__contains__("(256, 256)"):
            img_size = (256, 256)

        x_imgs = load_all_imgs(x_dir, img_size, 1)

        model_path = os.path.join(v2_result_path, folder, "result.h5")
        dependencies = {
            metric_name: mean_iou_threshold,
        }
        model = keras.models.load_model(model_path, custom_objects=dependencies)

        result_imgs = model.predict(x_imgs)

        imgs_paths = files = os.listdir(x_dir)
        imgs_paths.sort()

        for i in range(len(x_imgs)):
            img_name = imgs_paths[i].replace(".jpg", "")

            y_img_path = os.path.join(y_dir, img_name + ".png")
            y_img = cv2.imread(y_img_path, 0)
            y_img = np.expand_dims(y_img, 2)
            y_img = cv2.resize(y_img, img_size)

            if folder.__contains__("multiclass"):
                y_imgs_normalized = np.zeros(y_img.shape + (3,))

                y_imgs_normalized[y_img == 0] = np.array([1, 0, 0])
                y_imgs_normalized[y_img == 1] = np.array([0, 1, 0])
                y_imgs_normalized[y_img == 2] = np.array([0, 0, 1])
                y_img = y_imgs_normalized
                y_img = y_img * 255.0


            new_x_img_path = os.path.join(new_img_path, img_name + "_x.jpg")
            print("writing:", new_x_img_path)
            x_img = x_imgs[i]
            x_img = x_img * 255.0
            cv2.imwrite(new_x_img_path, x_img)

            new_result_img_path = os.path.join(new_img_path, img_name + "_result.png")
            print("writing:", new_result_img_path)
            result_img = result_imgs[i]
            result_img[result_img > threshold] = 1
            result_img[result_img <= threshold] = 0
            result_img = result_img * 255.0
            cv2.imwrite(new_result_img_path, result_img)

            new_y_img_path = os.path.join(new_img_path, img_name + "_y.png")
            print("writing:", new_y_img_path)
            cv2.imwrite(new_y_img_path, y_img)
