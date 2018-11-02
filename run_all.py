#!./weka_env/bin/python3
import os
import sys
import csv_arff
import part3_attributeReduction
import extract_pixels
import classify
import threading

csv_folder="fer2018/csv"
arff_folder="fer2018/arffs"
reduction_folder="fer2018/reduced_arffs"
transformed_arffs="fer018/transformed_arffs"
pixel_values="pixel_values"

preprocess = False
if(len(sys.argv)==2):
    if sys.argv[1]=="--preprocess":
        preprocess=True

filename="fer2018/transformed_arffs/transformed_70.arff"
testNum = "_70_pixels"

class myThread (threading.Thread):
   def __init__(self, threadID, name, function, args=None):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.function = function
      self.args = args
   def run(self):
      print ("Starting " + self.name)
      self.function(self.args)

def convert_to_arff():

    print("*** Converting CSVs ****\n")
    filenames = os.listdir(csv_folder)

    for csv in filenames:
        filename=os.path.join(csv_folder,csv)
        print("Converting "+filename)
        
        if(csv=="fer2018.csv"):
            csv_converter = csv_arff.Convert(filename,False)
            csv_converter.run()
        else:
            csv_converter = csv_arff.Convert(filename,True)
            csv_converter.run()

def reduce_attr():
    print("\n*** Reducing Arffs ****\n")
    filenames = os.listdir(arff_folder)

    for arff in filenames:
        filename=os.path.join(arff_folder,arff)
        print("Reducing "+filename)
        arff_reducer = part3_attributeReduction.reduce_attributes(filename)
        arff_reducer.run()

def extract():
    print("\n*** Extracting Pixels ****\n")
    filenames = os.listdir(pixel_values)

    for values in filenames:
        filename=os.path.join(pixel_values,values)
        print("Extracting from "+filename)
        extractor = extract_pixels.extract_pix(filename)
        extractor.run()

def run_ibk_crossval():
    global filename, testNum
    jvm_helper = classify.cw2_helper()

    ibkCls_crossval = classify.cw2_classifier()
    ibkCls_crossval.load_data(filename)
    ibkCls_crossval.run_ibk_crossval("results/test"+str(testNum))

def run_ibk_split():
    global filename, testNum
    jvm_helper = classify.cw2_helper()

    ibkCls = classify.cw2_classifier()
    ibkCls.load_data_split(filename,80)
    ibkCls.run_ibk_split("results/test"+str(testNum))

def run_nb_split():
    global filename, testNum
    jvm_helper = classify.cw2_helper()

    naiveBayesCls = classify.cw2_classifier()
    naiveBayesCls.load_data_split(filename,80)
    naiveBayesCls.run_naive_bayes_split("results/test"+str(testNum))

def run_nb_crossval():
    global filename, testNum
    jvm_helper = classify.cw2_helper()

    naiveBayesCls_crossval = classify.cw2_classifier()
    naiveBayesCls_crossval.load_data(filename)
    naiveBayesCls_crossval.run_naive_bayes_crossval("results/test"+str(testNum))

def run_bayes_split(parents=1):
    global filename, testNum
    jvm_helper = classify.cw2_helper()

    naiveBayesCls_crossval = classify.cw2_classifier()
    naiveBayesCls_crossval.load_data_split(filename,80)
    naiveBayesCls_crossval.run_bayes_split("results/test"+str(testNum),parents)

    naiveBayesCls_crossval = classify.cw2_classifier()
    naiveBayesCls_crossval.load_data_split(filename,80)
    naiveBayesCls_crossval.run_bayes_hill_split("results/test"+str(testNum),parents)

    naiveBayesCls_crossval = classify.cw2_classifier()
    naiveBayesCls_crossval.load_data_split(filename,80)
    naiveBayesCls_crossval.run_bayes_tan_split("results/test"+str(testNum),parents)

def run_classifiers():

    jvm_helper = classify.cw2_helper(False)
    jvm_helper.cleanup()

    threads = []

    #TASK 1
    # Create new threads
    # thread1 = myThread(1, "IBK-Cross-Val", run_ibk_crossval)
    # thread2 = myThread(2, "IBK-Split", run_ibk_split)
    # thread3 = myThread(3, "run_nb_split", run_nb_split)
    # thread4 = myThread(4, "run_nb_crossval", run_nb_crossval)

    #TASK2
    thread1 = myThread(1, "run_bayes_split", run_bayes_split, (1))
    thread2 = myThread(2, "run_bayes_split", run_bayes_split, (2))
    thread3 = myThread(3, "run_bayes_split", run_bayes_split, (3))

    # Start new Threads
    thread1.start()
    thread2.start()
    thread3.start()
    # thread4.start()

    # Add threads to thread list
    threads.append(thread1)
    threads.append(thread2)
    threads.append(thread3)
    #threads.append(thread4)

    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    jvm_helper.cleanup()
    print ("Exiting Main Thread")

try:
    if(preprocess):
        print("***** Preprocessing Data ******")
        convert_to_arff()
        reduce_attr()
        extract()
    else:
        run_classifiers()

except Exception as e:
    print(e)


    