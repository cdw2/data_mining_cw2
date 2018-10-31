import os
import sys
import traceback
import weka.core.jvm as jvm
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation

data_dir = "fer2018/arffs/"
tt_file = "fer2018.arff"

def main(args):
    data_file = data_dir + tt_file

    print("\nLoading dataset: " + data_file)
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(data_file)
    data.class_is_first()

    # generate train/test split of randomized data
    train, test = data.train_test_split(80.0, Random(1))

    print("\nBuilding Classifier on training data.")
    # build classifier
    cls = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
    cls.build_classifier(train)
    print(cls)

    print("\nEvaluating on test data.")
    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())


if __name__ == "__main__":
    try:
        jvm.start()
        main(sys.argv)
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()