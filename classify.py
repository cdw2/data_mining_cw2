import os
import sys
import traceback
import weka.core.jvm as jvm
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation
from datetime import datetime

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
        buildTimeStart=datetime.now()
        cls = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
        cls.build_classifier(self.training_data)

        resultsString = ""
        resultsString += self.print_both(str(cls),resultsString)

        buildTimeString = "Classifier Built in "+str(datetime.now()-buildTimeStart)+" secs.\n"
        resultsString += self.print_both(buildTimeString,resultsString)
        
        #Evaluate Classifier
        resultsString += self.print_both("\nEvaluating on test data.",resultsString)

        buildTimeStart=datetime.now()
        evl=Evaluation(self.training_data)
        evl.test_model(cls, self.testing_data)

        resultsString += self.print_both(str(evl.summary()),resultsString)
        buildTimeString = "Classifier Evaluated in "+str(datetime.now()-buildTimeStart)+" secs.\n"
        resultsString += self.print_both(buildTimeString,resultsString)
        

        #Save Results and Cleanup
        self.save_results("Naive_Bayes",resultsString,output_directory)
        self.cleanup()

    def save_results(self, classifier, string, output_directory):
        try:
            os.mkdir(output_directory)
        except:
            print("Directory Exists, Continuting.\n")
        
        output_file_path = os.path.join(output_directory,classifier+"results.txt")

        try:
            output_file = open(output_file_path,"x")
        except:
            os.remove(output_file_path)
            print("Removed exisiting file\n")
            output_file = open(output_file_path,"a")

        output_file.write(string) 
        output_file.close()
        print("\nResults saved to :"+output_file_path)


    def cleanup(self):
        jvm.stop()

    def print_both(self,print_string, resultsString):
        print(print_string)
        resultsString += print_string
        return resultsString
