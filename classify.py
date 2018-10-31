import os
import sys
import traceback
import weka.core.jvm as jvm
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation

class classify():
    def __init__(self, filename, validation_split):
        jvm.start()
        self.filename = filename
        self.validation_split = validation_split
        self.load_data(filename)       

    def load_data(self, filename):
        print("\nLoading dataset: " + filename)
        loader = Loader(classname="weka.core.converters.ArffLoader")
        data = loader.load_file(filename)
        data.class_is_first()
        train, test = data.train_test_split(self.validation_split, Random(1))
        self.training_data = train
        self.testing_data = test

    def run_naive_bayes(self):
        print("\nBuilding Classifier on training data.")
        # build classifier
        cls = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
        cls.build_classifier(self.training_data)
        print(cls)

        print("\nEvaluating on test data.")
        # evaluate
        evl = Evaluation(self.training_data)
        evl.test_model(cls, self.testing_data)
        print(evl.summary())

    def cleanup(self):
        jvm.stop()