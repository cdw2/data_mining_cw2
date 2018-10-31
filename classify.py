import os
import sys
import traceback
import weka.core.jvm as jvm
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation

class classifier():
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

    def run_naive_bayes(self, output_directory):
        
        # build classifier
        print("\nBuilding Classifier on training data.")
        cls = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
        cls.build_classifier(self.training_data)
        print(cls)

        resultsString = str(cls)

        #Evaluate Classifier
        print("\nEvaluating on test data.")
        resultsString+="\nEvaluating on test Data:\n"
        evl=Evaluation(self.training_data)
        evl.test_model(cls, self.testing_data)
        print(evl.summary())
        resultsString+=str(evl.summary())

        #Save Results and Cleanup
        self.save_results("Naive_Bayes",resultsString,output_directory)
        self.cleanup()

    def save_results(self, classifier, string, output_directory):
        try:
            os.mkdir(output_directory)
        except:
            print("Directory Exists, Continuting.\n")
        
        output_file = os.path.join(output_directory,classifier+"results.txt")

        try:
            output_file = open(output_file,"x")
        except:
            os.remove(output_file)
            print("Removed exisiting file\n")
            output_file = open(output_file,"a")

        output_file.write(string) 
        output_file.close() 


    def cleanup(self):
        jvm.stop()