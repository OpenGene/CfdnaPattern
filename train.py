import sys, os
import fastq
from optparse import OptionParser
from multiprocessing import Process, Queue
import time
from util import *
from feature import *
import numpy as np
from sklearn import svm

def parseCommand():
    usage = "extract the features, and train the model, from the training set of fastq files. \n\npython training.py <fastq_files> [-f feature_file] [-m model_file] "
    version = "0.0.1"
    parser = OptionParser(usage = usage, version = version)
    parser.add_option("-f", "--feature", dest = "feature_file", default = "features.json",
        help = "specify which file to store the extracted features from training set.")
    parser.add_option("-m", "--model", dest = "model_file", default = "cfdna.model",
        help = "specify which file to store the built model.")
    return parser.parse_args()

def main():
    time1 = time.time()
    if sys.version_info.major >2:
        print('python3 is not supported yet, please use python2')
        sys.exit(1)

    data = []
    label = []
    fq_files = get_arg_files()
    for fq in fq_files:
        extractor = FeatureExtractor(fq)
        extractor.extract()
        feature = extractor.feature()
        if "cfdna" in fq.lower():
            data.append(feature)
            label.append(1)
        elif "gdna" in fq.lower() or "ffpe" in fq.lower():
            data.append(feature)
            label.append(0)

        print("")
        print(fq)
        print(feature)

    print("start training...")
    clf = svm.LinearSVC()
    clf.fit(np.array(data), np.array(label))
    print(clf.score(np.array(data), np.array(label)))
    
    (options, args) = parseCommand()
    time2 = time.time()
    print('Time used: ' + str(time2-time1))

if __name__  == "__main__":
    main()