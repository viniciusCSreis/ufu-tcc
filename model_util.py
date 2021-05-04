import csv
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


def load_all_imgs(path, image_size, mode):
    imgs = []
    files = os.listdir(path)
    files.sort()
    print(f'\n{path} found {len(files)} img to load')
    i = 0
    for file in files:
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path, mode)
        if mode == 0:
            img = np.expand_dims(img, 2)
        img = cv2.resize(img, image_size)
        img = img / float(255)
        imgs.append(img)
        if i % 100 == 0:
            print(f'\n[{i}]:', end='')
        else:
            print("|", end='')
        i = i + 1
    return np.array(imgs)


class Saver:

    def __init__(self, model, history, dir_path):
        self.model = model
        self.history = history
        self.dir_path = dir_path

    def save_model(self):
        filename = model_path(self.dir_path)
        self.model.save(filename)
        print("saved to:", filename)

    def save_metrics(self, metric_name):
        plt.plot(self.history.history[str(metric_name)])
        if 'val_' + metric_name in self.history.history.keys():
            plt.plot(self.history.history['val_' + metric_name])
        plt.title("Acurácia por épocas")
        plt.xlabel("épocas")
        plt.legend(['treino', 'validação'])
        output_path = os.path.join(self.dir_path, "acc_por_epocas")
        plt.savefig(output_path)
        print("saved to:", output_path)

    def save_loss(self):
        plt.plot(self.history.history['loss'])
        if 'val_loss' in self.history.history.keys():
            plt.plot(self.history.history['val_loss'])
        plt.title("Perda por épocas")
        plt.xlabel("épocas")
        plt.legend(['treino', 'validação'])
        output_path = os.path.join(self.dir_path, "loss_por_epocas")
        print("saved to:", output_path)
        plt.savefig(output_path)


class ImgTester:
    def __init__(self, dir_path, train_img_dir, test_img_dir, img_size):
        self.dir_path = dir_path
        self.train_img_dir = train_img_dir
        self.test_img_dir = test_img_dir
        self.img_size = img_size

    def test(self, img_to_test, img_to_plot, metrics, threshold, dependencies):

        model = keras.models.load_model(model_path(self.dir_path), custom_objects=dependencies)

        i = 0
        n = 1

        plt.figure(figsize=(10, 10))

        files = os.listdir(self.train_img_dir)
        print(f'found {len(files)} img to test')

        predict_metrics = []

        for file in files:
            train_path = os.path.join(self.train_img_dir, file)
            test_path = os.path.join(self.test_img_dir, file)
            test_path = test_path.replace(".jpg", ".png")

            train_img = cv2.imread(train_path, 1)
            original_train_img = train_img
            train_img_shape = train_img.shape
            train_img = cv2.resize(train_img, self.img_size)
            train_img = train_img / float(255)

            test_img = cv2.imread(test_path, 0)
            original_test_img = test_img
            test_img = cv2.resize(test_img, self.img_size)
            test_img = np.expand_dims(test_img, 2)
            test_img = test_img / float(255)

            predict_images = model.predict(np.array([train_img]))
            predict_img = predict_images[0]
            predict_img[predict_img > threshold] = 1
            predict_img[predict_img <= threshold] = 0

            result_metrics = []
            for metric in metrics:
                metric.reset_states()
                metric.update_state(test_img, predict_img)
                predict_metric = metric.result().numpy()
                result_metrics.append(predict_metric)

            predict_metrics.append({"name": file, "metric": result_metrics})

            predict_img_original_size = cv2.resize(
                predict_img,
                (train_img_shape[1], train_img_shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            predict_img_original_size = predict_img_original_size * 255.0

            if i < img_to_plot:
                for j in range(3):
                    plt.xticks([])
                    plt.yticks([])
                    plt.grid(False)
                    plt.subplot(img_to_plot, 3, n)
                    if j == 0:
                        plt.imshow(train_img)
                        plt.xlabel("Original " + file)
                    if j == 1:
                        plt.imshow(predict_img, cmap=plt.cm.binary)
                        plt.xlabel("Result", )
                    if j == 2:
                        plt.imshow(test_img, cmap=plt.cm.binary)
                        plt.xlabel("Expected")
                    n = n + 1

            result_dir = os.path.join(self.dir_path, "result-images")
            try:
                os.mkdir(result_dir)
            except OSError:
                pass

            result_filename = file.replace(".jpg", "_result.png")
            result_path = os.path.join(result_dir, result_filename)
            cv2.imwrite(result_path, predict_img_original_size)

            x_filename = file.replace(".jpg", "_x.jpg")
            x_path = os.path.join(result_dir, x_filename)
            x = original_train_img
            cv2.imwrite(x_path, x)

            y_filename = file.replace(".jpg", "_y.png")
            y_path = os.path.join(result_dir, y_filename)
            y = original_test_img
            cv2.imwrite(y_path, y)

            if 0 < img_to_test <= i:
                break

            if i % 100 == 0:
                print(f'\n[{i}]:', end='')
            else:
                print("|", end='')
            i = i + 1

        plt.show()
        predict_metrics_path =os.path.join(self.dir_path, 'metrics.csv')
        wtr = csv.writer(open(predict_metrics_path, 'w'), delimiter=',', lineterminator='\n')
        for p_metric in predict_metrics:
            row = [p_metric['name']]
            row.extend(p_metric['metric'])
            wtr.writerow(row)
        print("\nWrite metrics to:", predict_metrics_path)


def model_path(base_path):
    return os.path.join(base_path, "result.h5")
