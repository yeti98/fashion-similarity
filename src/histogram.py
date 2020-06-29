import os
import pickle

import cv2 as cv
from imutils import paths


def save_histogram(histogram_dict, name):
    assert name != None
    file = open(name, 'wb')
    pickle.dump(histogram_dict, file)
    file.close()


def calc_images_histogram(folder):
    image_paths = list(paths.list_images(folder))
    images = {}
    histogram_dict = {}
    for (idx, imagePath) in enumerate(image_paths):
        # extract the image filename (assumed to be unique) and
        # load the image, updating the images dictionary
        file_name = imagePath.split("/")[-1]
        img = cv.imread(imagePath, 1)
        images[file_name] = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # extract a 3D RGB color histogram from the image,
        # using 8 bins per channel, normalize, and update
        # the histogram_dict
        hist = cv.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv.normalize(hist, hist).flatten()
        histogram_dict[file_name] = hist

    return histogram_dict


def load_histogram_items(histogram_path):
    file = open(histogram_path, 'rb')
    hdict = pickle.load(file)
    file.close()
    return hdict


if __name__ == '__main__':
    prefix = '../fashion_data'
    model_prefix = '../model'
    for (idx, label) in enumerate(os.listdir(prefix)):
        histogram_dict = calc_images_histogram('{}/{}'.format(prefix, label))
        save_histogram(histogram_dict, '{}/{}.pkl'.format(model_prefix, label))
    pass
