import os
import cv2
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


def generate_internal_external_img(external_path, internal_path, result_path):
    files = os.listdir(internal_path)
    files.sort()
    print(f'\n{internal_path} found {len(files)} img to load')
    i = 0
    for file in files:
        external_img_path = os.path.join(external_path, file)
        internal_img_path = os.path.join(internal_path, file)
        result_img_path = os.path.join(result_path, file)
        external_img = cv2.imread(external_img_path, 0)
        internal_img = cv2.imread(internal_img_path, 0)
        result_img = np.zeros(internal_img.shape + (3,))
        result_img[external_img == 255] = 1
        result_img[internal_img == 255] = 2
        cv2.imwrite(result_img_path, result_img)


if __name__ == '__main__':
    generate_internal_external_img(
        "../../imagens_cra/validation/cra",
        "../../imagens_cra/validation_interna/cra",
        "../../imagens_cra/validation_interna_externa_v2/cra"
    )
    print("finish !")
