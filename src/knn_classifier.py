from __future__ import absolute_import, division, print_function, unicode_literals

import os

import cv2 as cv

K = 5

# TensorFlow and tf.keras
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from imutils import paths

label_class_mapper = {0: 'Áo thun', 1: 'Quần dài', 2: 'Áo len', 3: 'Váy', 4: 'Áo khoác', 5: 'Sandal', 6: 'Áo sơ mi',
                      7: 'Giày',
                      8: 'Túi xách', 9: 'Ủng'}
class_label_mapper = dict(zip(label_class_mapper.values(), label_class_mapper.keys()))

accept_class = ['Áo thun', 'Quần dài', 'Váy', 'Áo sơ mi']


def load_fashion_mnist_dataset():
    fashion_mnist = keras.datasets.fashion_mnist
    return fashion_mnist.load_data()


def reduce_on_array(images, labels, accept_classes):
    reduced_train = []
    reduced_labels = []
    for index in range(images.shape[0]):
        current_labels = labels[index]
        if current_labels in [class_label_mapper[x] for x in accept_classes]:
            reduced_train.append(images[index])
            reduced_labels.append(current_labels)
    return (np.array(reduced_train), np.array(reduced_labels))


def dataset_filter(accept_classes):
    (images, labels), (test_images, test_labels) = load_fashion_mnist_dataset()
    return reduce_on_array(images, labels, accept_classes), reduce_on_array(test_images, test_labels, accept_classes)


def visual_sample_images(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(0, 25, 1):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(label_class_mapper[labels[i]])
    plt.show()


# visual_sample_images(train_images, train_labels)
# visual_sample_images(test_images, test_labels)

def train(train_images, train_labels):
    print("[INFO] Training k-NN classifier")
    model = KNeighborsClassifier(n_neighbors=K)
    train_len = train_images.shape[0]
    train_data = train_images.reshape((train_len, 28 * 28))

    model.fit(train_data, train_labels)
    return model


def test(model, test_images, test_labels):
    print("[INFO] Evaluating k-NN classifier")
    from sklearn.metrics import classification_report
    test_len = test_images.shape[0]
    test_data = test_images.reshape((test_len, 28 * 28))
    print(classification_report(test_labels, model.predict(test_data)))


def save_model(model):
    import pickle
    pickle.dump(
        model, open("../model/fashion_model_" + str(model.n_neighbors) + ".pkl", "wb")
    )


def load_knn_model(name):
    import pickle
    file = open(name, 'rb')
    model = pickle.load(file)
    file.close()
    return model


def predict(model, image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(image, 240, 255, cv.THRESH_BINARY)
    image[thresh == 255] = 0

    # Resize image
    image = cv.resize(image, (28, 28))

    test_real = np.array([image])
    test_real = test_real / 255.0

    z = test_real.reshape(1, -1)

    label_index = model.predict(z)[0]

    print(model.predict_proba(z))

    return label_index, label_class_mapper[label_index]


if __name__ == '__main__':
    # PREPARE
    # (train_images, train_labels), (test_images, test_labels) = dataset_filter(accept_class)
    # train_images = train_images / 255.0
    # test_images = test_images / 255.0

    # TRAIN
    # knn_model = train(train_images, train_labels)
    # save_model(knn_model)

    # TEST
    # knn_model = load_knn_model(name='../model/fashion_model_5.pkl')
    # test(knn_model, test_images, test_labels)

    # PREDICT
    # knn_model = load_knn_model(name='../model/fashion_model_5.pkl')
    # plt.figure(figsize=(10, 10))
    # test_len = test_images.shape[0]
    # test_data = test_images.reshape((test_len, 28 * 28))
    # for i in range(0, 25, 1):
    #     label_index = knn_model.predict([test_data[i]])[0]
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(test_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(label_class_mapper[label_index])
    # plt.show()

    # TEST ONE IMAGE
    # knn_model = load_knn_model(name='../model/fashion_model_5.pkl')

    # trouser_image = cv.imread('../fashion_data/trouser/trouser.jpg')
    # label_index = predict(knn_model, trouser_image)
    # print(label_class_mapper[label_index])

    # TEST FOLDER
    knn_model = load_knn_model(name='../model/fashion_model_5.pkl')
    image_paths = list(paths.list_images("../fashion_data"))

    for (idx, path) in enumerate(image_paths):
        # Load images
        # Assuming path in following format:
        # /path/to/dataset/{class}/{image-name}.jpg
        label = path.split(os.path.sep)[-2]

        image = cv.imread(path)

        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        ret, thresh = cv.threshold(image, 240, 255, cv.THRESH_BINARY)
        #
        image[thresh == 255] = 0

        # Resize image
        image = cv.resize(image, (28, 28))

        test_real = np.array([image])
        test_real = test_real / 255.0

        z = test_real.reshape(1, -1)

        label_index = knn_model.predict([z[0]])[0]
        prob = knn_model.predict_proba([z[0]])
        print("***********")
        print(prob)

        plt.figure()
        plt.imshow(test_real[0], cmap=plt.cm.binary)
        plt.colorbar()
        plt.grid(False)
        plt.xlabel(label + "->" + label_class_mapper[label_index])
        plt.show()

    pass
