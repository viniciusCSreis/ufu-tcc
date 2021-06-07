import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
import csv

if __name__ == '__main__':

    metric = tf.keras.metrics.MeanIoU(num_classes=2)

    v2_result_path = '../imagens_cra/result/v2'
    v2_folders = os.listdir(v2_result_path)
    v2_folders = list(filter(lambda f: str(f).startswith("unet_multiclass"), v2_folders))
    for folder in v2_folders:
        img_path = os.path.join(v2_result_path, folder, "result-images-train-size")

        print("img_path:", img_path)

        imgs = pd.unique(list(
            map(
                lambda img: str(img)
                    .replace("_result.png", "")
                    .replace("_x.jpg", "")
                    .replace("_y.png", "")
                , os.listdir(img_path)
            )
        )).tolist()
        imgs.sort()

        metrics = [{
            "name": "imgs",
            "metrics": "IoU",
        }]

        for img in imgs:
            result_img_path = os.path.join(img_path, img + "_result.png")

            y_img_path = os.path.join(img_path, img + "_y.png")

            if folder.__contains__("multiclass"):
                result_img = cv2.imread(result_img_path, 1)
            else:
                result_img = cv2.imread(result_img_path, 0)
                result_img = np.expand_dims(result_img, 2)

            if folder.__contains__("multiclass"):
                y_img = cv2.imread(y_img_path, 1)
            else:
                y_img = cv2.imread(y_img_path, 0)
                y_img = np.expand_dims(y_img, 2)

            metric.reset_states()
            result_img = result_img / 255.0
            y_img = y_img / 255.0
            metric.update_state(y_img, result_img)
            predict_metric = metric.result().numpy()
            metrics.append({
                "name": img,
                "metrics": predict_metric
            })

        predict_metrics_path = os.path.join(v2_result_path, folder, 'metrics_train-size.csv')
        f = open(predict_metrics_path, 'w')
        wtr = csv.writer(f, delimiter=',', lineterminator='\n')
        for p_metric in metrics:
            row = [p_metric['name'], p_metric['metrics']]
            wtr.writerow(row)

        f.close()
        print("finish.")
