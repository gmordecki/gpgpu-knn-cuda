import gzip
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report

import regex
np.set_printoptions(threshold=np.inf, precision=5)

sizes = [100, 500, 1000, 5000, 10000]
dimensions = [100, 500, 1000, 5000, 10000]
test_size = 1000
DATASET_PATH = 'dataset'
classes = [
    'Books',
    'Clothing_Shoes_and_Jewelry',
    'Electronics',
    'Home_and_Kitchen',
]


def parse(path):
  g = gzip.open(f'{DATASET_PATH}/reviews_{path}_5.json.gz', 'r')
  for l in g:
    yield eval(l)


def get_n(iterator, n):
    reviews = []

    for i in range(n):
        review = next(iterator)
        reviews.append(review['summary'] + ' ' + review['reviewText'])
    return reviews


def save_to_file(dimension, size, X, y, name):
    to_save = X.todense()

    to_save = np.hstack((np.atleast_2d(y).T, to_save))
    print(f"{DATASET_PATH}/{name}_reviews_{dimension}_{size}.csv tamaño {to_save.shape}")

    np.savetxt(f"{DATASET_PATH}/{name}_reviews_{dimension}_{size}.csv", to_save, fmt='%i' + ',%1.5f' * (to_save.shape[1] - 1))


def classify_test(X_train_tf_idf, X_test_tf_idf):
    knn = KNeighborsClassifier(n_neighbors=5)
    clf = knn.fit(X_train_tf_idf, [t[1] for t in train_reviews])
    y_pred = knn.predict(X_test_tf_idf)
    distances, indices = knn.kneighbors(X_test_tf_idf)
    # print(distances)
    # print(indices)

    print(classification_report([t[1] for t in test_reviews], y_pred))


classify = True

if __name__ == "__main__":
    for i, size in enumerate(sizes):
        print(f'Generando file de {size}')
        train_reviews, test_reviews = [], []
        for cl in classes:
            iterator = parse(cl)
            train_reviews += [(t, cl) for t in get_n(iterator, round(size / len(classes)))]
            test_reviews += [(t, cl) for t in get_n(iterator, round(test_size / len(classes)))]

        for dimension in dimensions:
            print(f'-------- dim {dimension} - num train {len(train_reviews)}')
            vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=0, max_features=dimension)
            X_train_tf_idf = vectorizer.fit_transform([t[0] for t in train_reviews])

            X_test_tf_idf = vectorizer.transform([t[0] for t in test_reviews])

            if classify:
                classify_test(X_train_tf_idf, X_test_tf_idf)

            save_to_file(dimension, size, X_train_tf_idf, [classes.index(t[1]) for t in train_reviews], "train")

            X_test_tf_idf = vectorizer.transform([t[0] for t in test_reviews])

            save_to_file(dimension, size, X_test_tf_idf, [classes.index(t[1]) for t in test_reviews], "test")
