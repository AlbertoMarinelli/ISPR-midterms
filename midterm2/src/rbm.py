import math
import copy
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from tqdm import tqdm
from functions import sigmoid
from sklearn.metrics import confusion_matrix
from load_dataset import load_data


class RBM ():

    def __init__(self, v_units, h_units=1, weights=None, v_bias=None, h_bias=None, path_directory=None):

        self.tr_imgs, self.tr_labels, self.ts_imgs, self.ts_labels = load_data(path_directory)

        self.v_units = v_units # Number of visible units
        self.h_units = h_units # Number of hidden units

        self.weights = weights if weights is not None else np.random.uniform(-1, 1, size=(h_units, v_units)) # Weights matrix
        self.v_bias = v_bias if v_bias is not None else np.zeros(v_units) # Bias vector for visible layer
        self.h_bias = h_bias if h_bias is not None else np.zeros(h_units) # Bias vector for hidden layer

        self.classifier = tf.keras.models.Sequential([  # Set classifier
            Dense(units=200, activation='relu', input_dim=h_units),
            Dense(units=100, activation='relu'),
            Dense(units=50, activation='relu'),
            Dense(units=10, activation='softmax')
        ])

    def hidden_visible(self, v_sample):
        h_probs = sigmoid(np.add(np.matmul(self.weights, v_sample), self.h_bias))
        h_samples = np.random.binomial(n=1, p=h_probs, size=len(h_probs))
        return h_probs, h_samples

    def visible_hidden(self, h_sample):
        v_probs = sigmoid(np.add(np.matmul(h_sample, self.weights), self.v_bias))
        v_samples = np.random.binomial(n=1, p=v_probs, size=len(v_probs))
        return v_probs, v_samples

    def gibbs_sampling(self, h_sample, k):
        v_prob, v_sample, h_prob = None, None, None
        for i in range(k):
            v_prob, v_sample = self.visible_hidden(h_sample)
            h_prob, h_sample = self.hidden_visible(v_sample)
        return v_prob, v_sample, h_prob, h_sample

    def contrastive_divergence(self, v_probs, k):
        # COMPUTE WAKE
        v_sample = np.random.binomial(n=1, p=v_probs, size=len(v_probs))
        h_probs, h_sample = self.hidden_visible(v_sample)
        wake = np.outer(h_probs, v_probs)

        # COMPUTE DREAM
        v_probs_gibbs, v_sample_gibbs, h_probs_gibbs, h_sample_gibbs = self.gibbs_sampling(h_sample, k)
        dream = np.outer(h_probs_gibbs, v_probs_gibbs)

        deltaW = np.subtract(wake, dream)
        deltaBv = np.subtract(v_sample, v_sample_gibbs)
        deltaBh = np.subtract(h_sample, h_sample_gibbs)

        return deltaW, deltaBv, deltaBh

    def encode(self, images=None): # Encodings for the images to feed the classifier
        images = self.tr_imgs if images == 'train' else (self.ts_imgs if images == 'test' else images)
        encodings = []
        for i in range(len(images)):
            v_sample = np.random.binomial(n=1, p=images[i], size=len(images[i]))
            probs = [images[i]] + [None] * (1)
            samples = [v_sample] + [None] * (1)
            probs[1] = sigmoid(np.add(np.matmul(self.weights, samples[0]), self.h_bias))
            samples[1] = np.random.binomial(n=1, p=probs[1], size=len(probs[1]))
            enc = samples
            encodings.append(enc[-1])
        return encodings

    def fit_classifier(self, load_boltz_weights=False, w_path=None, save=False):
        if load_boltz_weights:
            self.load_weights(w_path)

        # create a training set by encoding all the training images
        tr_set = self.encode('train')

        # 1-hot encoding of the labels
        train_labels = tf.stack(to_categorical(self.tr_labels, 10))
        tr_set = tf.stack(tr_set)

        # compile and fit the classifier
        self.classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
        hist = self.classifier.fit(x=tr_set, y=train_labels, epochs=5)

        # save the classifier's weights
        if save:
            save_path = datetime.now().strftime("classifier_%d-%m-%y_%H-%M.h5")
            self.classifier.save_weights(save_path)
        return hist

    def fit(self, epochs=1, lr=0.1, k=1, batch_size=1, save=False, fit_cl=False, save_weights=False):
        n_imgs = len(self.tr_labels)
        batch_size = n_imgs if batch_size == 'batch' else batch_size
        disable_tqdm = (False, True) if batch_size < n_imgs else (True, False)
        indexes = list(range(len(self.tr_imgs)))
        for ep in range(epochs):
            np.random.shuffle(indexes)
            self.tr_imgs = self.tr_imgs[indexes]
            self.tr_labels = self.tr_labels[indexes]

            for batch_idx in tqdm(range(math.ceil(len(self.tr_labels) / batch_size)), disable=disable_tqdm[0]): # Iterate on batches
                delta_W = np.zeros(shape=self.weights.shape)
                delta_bv = np.zeros(shape=(len(self.v_bias),))
                delta_bh = np.zeros(shape=(len(self.h_bias),))
                start = batch_idx * batch_size
                end = start + batch_size
                batch_imgs = self.tr_imgs[start: end]

                for img in tqdm(batch_imgs, disable=disable_tqdm[1]): # Iterate on patterns within a batch
                    dW, dbv, dbh = self.contrastive_divergence(v_probs=img, k=k)
                    delta_W = np.add(delta_W, dW)
                    delta_bv = np.add(delta_bv, dbv)
                    delta_bh = np.add(delta_bh, dbh)
                # UPDATE
                self.weights += (lr / batch_size) * delta_W
                self.v_bias += (lr / batch_size) * delta_bv
                self.h_bias += (lr / batch_size) * delta_bh
        if save:
            self.save_model(datetime.now().strftime("model_%d-%m-%y_%H-%M"))
        if fit_cl:
            # train the classifier on the embeddings
            self.fit_classifier(save=save_weights)

    def test_classifier(self, test_images=None, test_labels=None):
        # if a specific set of test images and labels is NOT specified, use the MNIST test set
        if test_images is not None and test_labels is not None:
            assert len(test_images) == len(test_labels)
        elif not (test_images is None and test_labels is None):
            raise RuntimeWarning("The number of test images differs from the number of test labels!")
        if test_images is None:
            test_images = self.ts_imgs
            test_labels = copy.deepcopy(self.ts_labels)

        # create a test set by encoding all the test images
        test_encodings = self.encode('test')

        # 1-hot encoding of the test labels
        test_encodings = tf.stack(test_encodings)
        test_labels = tf.stack(to_categorical(test_labels))

        # classifier evaluation
        results = self.classifier.evaluate(x=test_encodings, y=test_labels, return_dict=True)

        return results

    def confusion_matrix(self, test_images=None, test_labels=None):
        # if a specific set of test images and labels is NOT specified, use the MNIST test set
        if test_images is not None and test_labels is not None:
            assert len(test_images) == len(test_labels)
        elif not (test_images is None and test_labels is None):
            raise RuntimeWarning(
                "The number of test images differs from the number of test labels!")
        if test_images is None:
            test_images = self.ts_imgs
            test_labels = copy.deepcopy(self.ts_labels)

        # create a test set by encoding all the test images
        test_encodings = []
        for i in range(len(test_images)):
            v_sample = np.random.binomial(n=1, p=test_images[i], size=len(test_images[i]))
            probs = [test_images[i]] + [None] * (1)
            samples = [v_sample] + [None] * (1)
            probs[1] = sigmoid(np.add(np.matmul(self.weights, samples[0]), self.h_bias))
            samples[1] = np.random.binomial(n=1, p=probs[1], size=len(probs[1]))
            encoding = samples
            test_encodings.append(encoding[-1])
        predictions = self.classifier.predict(tf.stack(test_encodings))
        predictions = np.argmax(predictions, axis=1)
        conf_matr = confusion_matrix(y_true=test_labels, y_pred=predictions)
        plt.figure(figsize=(7, 5))
        sns.heatmap(conf_matr, annot=True, annot_kws={'size': 8})
        plt.xlabel('True labels')
        plt.ylabel('Predictions')
        plt.title('Confusion matrix')
        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        path = path if (path.endswith('.pickle') or path.endswith('.pkl')) else path + '.pickle'
        dump_dict = {
                     'weights_matrices': self.weights,
                     'v_bias': self.v_bias,
                     'h_bias': self.h_bias}
        with open(path, 'wb') as f:
            pickle.dump(dump_dict, f)

    def load_weights(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            weights_matrices = data['weights_matrices']
            v_bias = data['v_bias']
            h_bias = data['h_bias']
            self.weights = weights_matrices
            self.v_bias = v_bias
            self.h_bias= h_bias

    def show_reconstruction(self, img):
        v_sample = np.random.binomial(n=1, p=img, size=len(img))
        h_probs, h_sample = self.hidden_visible(v_sample)
        v_probs, v_sample = self.visible_hidden(h_sample)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.reshape(img, newshape=(28, 28)))
        ax[0].set_title('Original image')
        ax[1].imshow(np.reshape(v_probs, newshape=(28, 28)))
        ax[1].set_title('Reconstructed image')
        fig.suptitle('Reconstruction')
        fig.tight_layout()
        fig.show()