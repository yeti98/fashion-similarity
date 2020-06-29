import cv2 as cv

from src.histogram import load_histogram_items
from src.knn_classifier import load_knn_model, predict


def hu_moments(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    feature = cv.HuMoments(cv.moments(image)).flatten()
    return feature


def square_rooted(x):
    """ return 3 rounded square rooted value """

    return round(sqrt(sum([a * a for a in x])), 3)


def cosine_similarity(x, y):
    """ return cosine similarity between two lists """

    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)

def histogram(image, histogram_dict):
    results = {}
    for (k, hist) in histogram_dict.items():
        current_histogram = cv.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        current_histogram = cv.normalize(current_histogram, current_histogram).flatten()
        d = cv.compareHist(current_histogram, hist, cv.HISTCMP_CHISQR)
        results[k] = d
        # sort the results
    results = sorted([(v, k) for (k, v) in results.items()], reverse=False)
    return results


def find_similar_images(img_path):
    image = cv.imread(img_path)

    # Classify
    knn_classifier = load_knn_model('../model/fashion_model_5.pkl')

    label_index, class_name = predict(knn_classifier, image)

    model_names = {'Váy': 'dress', 'Áo thun': 'shirt', 'Quần dài': 'trouser', 'Áo sơ mi': ''}
    print(label_index, class_name)

    # Calculate similarity
    histogram_dict = load_histogram_items('../model/{}.pkl'.format(model_names[class_name]))
    hu_feature = hu_moments(image)
    histogram_feature = histogram(image, histogram_dict)

    for (index, tpl) in enumerate(histogram_feature):
        value, file_name = tpl
        print(value)
        print(file_name)

    # print(histogram_feature)


if __name__ == '__main__':
    find_similar_images('dress.jpeg')