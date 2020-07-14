# coding: utf-8

import argparse
from logging import info, basicConfig, INFO
import logging
import os
import pickle
from datetime import datetime as dt

import pandas as pd
from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import wordpunct_tokenize

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV
from sklearn.svm import LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

LOG_HEAD = '[%(asctime)s] %(levelname)s: %(message)s'
basicConfig(format=LOG_HEAD, level=INFO)


class DataWorker:
    def __init__(self, embeddings_path="embeddings/arabic-news.bin",
                 train_data_path=None, test_data_path=None, binary_embeddings=True):
        self.embeddings, self.dimension = self.load_vec(embeddings_path, binary_embeddings)
        self.train_data = self.load_data(train_data_path)
        self.test_data = self.load_data(test_data_path)
        if not train_data_path and not test_data_path:
            logging.info("No data has been provided. Please provide at least a data source. Exiting...")
            exit(1)

    def load_vec(self, path, format):
        # vectors file
        """load the pre-trained embedding model"""
        if not self.check_file_exist(path):
            logging.info("{} Path to embeddings doesn't exist. Exiting...")
            exit(1)

        if format:
            w2v_model = KeyedVectors.load_word2vec_format(path, binary=True)
        else:
            w2v_model = KeyedVectors.load(path)

        w2v_model.init_sims(replace=True)  # to save memory
        vocab, vector_dim = w2v_model.syn0.shape
        return w2v_model, vector_dim

    def load_data(self, path):
        # TODO: update reading methods to handle multiple data sources
        if not self.check_file_exist(path):
            logging.info("{} Path to data doesn't exist. Skipping...")
            return None
        dataset = pd.read_csv(path)
        return dataset

    @staticmethod
    def check_file_exist(path):
        if path and os.path.isfile(path):
            return True
        return False


class SentimentAnalyzer(DataWorker):
    def __init__(self, embeddings_path, train_data_path, test_data, binary_embeddings=True):
        super().__init__(embeddings_path, train_data_path, test_data)
        self.classification_report = {}
        self.fit_models = {}
        self.predictions = {}

    @staticmethod
    def remove_nan(x):
        """remove NaN values from data vectors"""
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        x_clean = imp.fit_transform(x)
        return x_clean

    @staticmethod
    def tokenize(text):
        """
        :param text: a paragraph string
        :return: a list of words
        """

        try:
            try:
                txt = unicode(text, 'utf-8')  # py2
            except NameError:
                txt = text  # py3
            words = wordpunct_tokenize(txt)
            length = len(words)
        except TypeError:
            words, length = ['NA'], 0

        return words, length

    @staticmethod
    def calculate_metrics(y_test, pred, fold=None):
        classifier_accuracy = accuracy_score(y_test, pred)
        classifier_f1_score = f1_score(y_test, pred, pos_label=None, average='macro')
        classifier_precision = precision_score(y_test, pred)
        classifier_recall = recall_score(y_test, pred)

        if fold:
            logging.info("--------------Results for fold {} of cross validation--------------".format(fold))
        else:
            logging.info("------------------------------")
        logging.info("Accuracy: {}".format(classifier_accuracy))
        logging.info("F1 Score: {}".format(classifier_f1_score))
        logging.info("Precision: {}".format(classifier_precision))
        logging.info("Recall: {}".format(classifier_recall))
        logging.info("------------------------------")

        return [classifier_accuracy, classifier_f1_score, classifier_precision, classifier_recall]

    @staticmethod
    def load_pickle(path):
        try:
            with open(path, "rb") as pickle_file:
                return pickle.load(pickle_file)
        except FileNotFoundError:
            logging.info("Pickle file not found. Exiting...")
            exit(1)

    def feature(self, words):
        """average words' vectors"""

        feature_vec = np.zeros((self.dimension,), dtype="float32")
        retrieved_words = 0
        for token in words:
            try:
                feature_vec = np.add(feature_vec, self.embeddings[token])
                retrieved_words += 1
            except KeyError:
                pass  # if a word is not in the embeddings' vocabulary discard it

        np.seterr(divide='ignore', invalid='ignore')
        feature_vec = np.divide(feature_vec, retrieved_words)

        return feature_vec

    def tokenizer(self, text, type_='NaN'):
        tokens = []
        logging.info('Tokenizing the {} dataset ..'.format(type_))
        total_tokens = []
        for txt in text:
            words, num = self.tokenize(txt)
            tokens.append(words)
            total_tokens.append(num)
        logging.info(' ... total {} {} tokens.'.format(sum(total_tokens), type_))
        return tokens

    def average_feature_vectors(self, text_tokens, type_='NaN'):
        """
        :param text_tokens: a list of lists (each list contains words) e.g. [['hi','do'], ['you','see'], ... ]
        :param type_: (optional) type of examples text e.g. train / test
        :return: the average word vector of each list
        """
        feature_vectors = np.zeros((len(text_tokens), self.dimension), dtype="float32")
        logging.info("Vectorizing {} tokens ..".format(type_))
        for i, example in enumerate(text_tokens):
            feature_vectors[i] = self.feature(example)

        logging.info(" ... total {} {}".format(len(feature_vectors), type_))

        return feature_vectors

    def process_data(self, data):
        tokens = self.tokenizer(data)
        text_vectors = self.average_feature_vectors(tokens)
        text_vectors = self.remove_nan(text_vectors)
        return text_vectors

    def scoring(self, x_test, y_test=None, path_to_model=None):
        # load pickled models
        if path_to_model:
            logging.info("Loading Model")
            self.fit_models = self.load_pickle(path_to_model)
        # make predictions
        logging.info("Making predictions...")
        for model_name, fitted_model in self.fit_models.items():
            pred = fitted_model.predict(x_test)
            self.predictions.update({model_name: pred})
            logging.info("Generated predictions for {}".format(model_name))
            if y_test:
                scoring_metrics = self.calculate_metrics(pred, y_test)
                string = '{} | Accuracy. {:.2f}% F1. {:.2f}% Precision. {:.2f} Recall. {:.2f}'
                logging.info(string.format(model_name, scoring_metrics[0] * 100, scoring_metrics[1] * 100,
                                scoring_metrics[2] * 100, scoring_metrics[3] * 100))
            negative = len(pred[pred == 0])
            positive = len(pred[pred == 1])
            string = '{} | Positive Tweets. {:.2f}% Negative Tweets. {:.2f}% '
            logging.info(string.format(model_name, positive*100/pred.shape[0], negative*100/pred.shape[0]))
        return self.predictions

    def train_model(self, x_train, y_train, save_model=False, cross_val=False, kfolds=2):

        logging.info("Training classifiers...")

        # classifiers
        classifiers = [
            RandomForestClassifier(n_estimators=100),
            SGDClassifier(loss='log', penalty='l1'),
            LinearSVC(C=1e1),
            NuSVC(),
            LogisticRegressionCV(solver='liblinear'),
            GaussianNB(),
        ]

        # RUN classifiers
        for c in classifiers:
            classifier_name = c.__class__.__name__
            if cross_val and kfolds:
                self.cross_validate(c, x_train, y_train, kfolds)  # TODO: return predictions, plot ROC
            else:
                model = c.fit(x_train, y_train)
                logging.info("Finished training {}".format(classifier_name))
                self.fit_models.update({classifier_name: model})
                if save_model:
                    with open("./{}.sav".format(classifier_name), "wb") as file:
                        pickle.dump(model, file)

        return self.fit_models

    def cross_validate(self, classifier, X, y, kfolds):
        # cross validation
        kf = KFold(n_splits=kfolds)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[train_index]

            classifier.fit(X_train, y_train)
            pred = classifier.predict(X_test)
            _ = self.calculate_metrics(y_test, pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", help="path a pre-trained vectors model.")
    parser.add_argument("--train_dataset", help="path a labeled (0/1) sentiment dataset.", required=False)
    parser.add_argument("--score_dataset", help="path to data for sentiment analysis.", required=True)
    parser.add_argument("--load_models", help="path to load models", required=False)
    # parser.add_argument("--save_models", help="path to save models", required=False, default=False)
    # parser.add_argument("--cross_validation", help="flag to run cross validation", required=False, default=False)

    # pars args
    args = parser.parse_args()
    vec = args.vectors
    dataset_path = args.train_dataset #"datasets/mpqa-ar.csv"
    score_data = args.score_dataset
    load_model = args.load_models
    # save_models = args.save_models
    tag = dt.now().strftime("%Y_%m_%d_%H_%H_%M_%S")
    train_data = None
    test_data = None
    train_target = None
    test_target = None
    predictions = None
    user_id = None
    path_to_pickle = None
    arSA = SentimentAnalyzer(vec, dataset_path, score_data)

    if dataset_path:
        user_id = arSA.test_data['user_id']
        train_data = arSA.process_data(arSA.train_data['txt'])
        train_target = arSA.train_data['sentiment']

    if score_data:
        test_data = arSA.process_data(arSA.test_data['tweet'])
        try:
            test_target = arSA.test_data['sentiment']
            # train model
        except KeyError:
            logging.info("Target variable is not in the test data...")
            pass

        if not load_model:
            arSA.train_model(train_data, train_target)
            predictions = arSA.scoring(test_data, test_target)
        else:
            # scoring
            predictions = arSA.scoring(test_data, path_to_pickle)

    for model, pred in predictions.items():
        pred = pd.DataFrame(pred, columns=['sentiment'])
        output = pd.DataFrame({"ID": user_id, "tweet": arSA.test_data['tweet'], "sentiment": pred['sentiment']})
        output.to_csv("./output/" + model + "_" + tag + "_predictions.csv")

    logging.info("Process is complete...")
