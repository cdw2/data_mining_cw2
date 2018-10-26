import sys
import os.path
import random
import fileinput
import linecache

def createTestFile():
    # 80% is 28709.600
    
    
    
    f = open("Testshuffled.arff", "r")
    lines = f.readlines()
    outputFile = open("NTestshuffled.arff", 'w')

    i = 0
    seenList = []

    while(i <= 28709):
        outputFile.write(lines[i])
        i +=1
    
    f.close()


    filenames = ['header.arff', 'NTestshuffled.arff']

    with open("Test" + sys.argv[1], 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
                outfile.write("\n")
                text = "% \n"
                outfile.write(text * 3)
    f.close()


def createTrainingFile():
    #20% is 7177.4000
    f = open("Trainingshuffled.arff", "r")
    lines = f.readlines()
    outputFile = open("NTrainingshuffled.arff", 'w')
    
    i = 0
    seenList = []
    num_lines = sum(1 for line in open("TrainingShuffled.arff",'r'))

    while(i <= 7177):
        outputFile.write(lines[i])
        i +=1
    f.close()


    filenames = ['header.arff', 'NTrainingshuffled.arff']
    
    with open("Training" + sys.argv[1], 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
                outfile.write("\n")
                text = "% \n"
                outfile.write(text * 3)
    f.close()


def randomData():
    f = open("data.arff", "r")
    lines = f.readlines()
    outputFile = open("Testshuffled.arff", 'w')
    outputFileTwo = open("Trainingshuffled.arff", 'w')
    
    
    i = 0
    seenList = []
    num_lines = sum(1 for line in open("data.arff",'r'))
    while(i < num_lines):
        randomN = random.randint(1,num_lines-1)
        seenList.append(randomN)
        if randomN in seenList == True:
            while(randomN in seenList != True):
                randomN += 1
                seenList.append(randomN)
        else:
            i += 1
    for x in seenList:
        outputFile.write(lines[seenList[x]])

    i = 0
    seenList = []
    num_lines = sum(1 for line in open("data.arff",'r'))
    while(i < num_lines):
        randomN = random.randint(1,num_lines-1)
        seenList.append(randomN)
        if randomN in seenList == True:
            while(randomN in seenList != True):
                randomN += 1
                seenList.append(randomN)
        else:
                i += 1
    for x in seenList:
        outputFileTwo.write(lines[seenList[x]])

    f.close()


def createHeader():
    f = open(sys.argv[1],'r')
    copy = open("header.arff", "w")
    for line in f:
        if "DATA" in line:
            copy.write(line)
            break
        copy.write(line)
    f.close()

def createData():
    
    f = open(sys.argv[1],'r')
    copy = open("data.arff", "w")
    for line in f:
        if "DATA" in line:
            for line in f:
                copy.write(line)
    f.close()


def main():
    if (len(sys.argv) != 1):
        print("correct")
    else:
        print("Error...  you did not enter a filename")
        str = input("please enter an .arff file you wish to randomise: ")
        if(os.path.isfile(str) == True):
            sys.argv.append( "random"+str)
        else:
            while(os.path.isfile(str) != True):
                print ("file does not exist try again")
                str = input("please enter an .arff file you wish to randomise: ")


    createHeader()
    print("*** HEADER CREATED ***")
    createData()
    print("*** DATA CREATED ***")
    randomData()
    print("*** DATA RANDOMISED ***")
    createTestFile()
    print("*** NEW TEST CREATED ***")
    createTrainingFile()
    print("*** NEW TRAINING CREATED ***")

    os.remove('data.arff')
    os.remove('header.arff')
    os.remove('NTestshuffled.arff')
    os.remove('NTrainingshuffled.arff')
    os.remove('Testshuffled.arff')
    os.remove('Trainingshuffled.arff')



main()
