import sys, os
import fastq
from optparse import OptionParser
from multiprocessing import Process, Queue
import time
from util import *
from feature import *
import numpy as np
from sklearn import svm
import random

def parseCommand():
    usage = "extract the features, and train the model, from the training set of fastq files. \n\npython training.py <fastq_files> [-f feature_file] [-m model_file] "
    version = "0.0.1"
    parser = OptionParser(usage = usage, version = version)
    parser.add_option("-f", "--feature", dest = "feature_file", default = "features.json",
        help = "specify which file to store the extracted features from training set.")
    parser.add_option("-m", "--model", dest = "model_file", default = "cfdna.model",
        help = "specify which file to store the built model.")
    parser.add_option("-c", "--cfdna_flag", dest = "cfdna_flag", default = "cfdna",
        help = "specify the filename flag of cfdna files, separated by semicolon")
    parser.add_option("-o", "--other_flag", dest = "other_flag", default = "gdna;ffpe",
        help = "specify the filename flag of other files, separated by semicolon.")
    parser.add_option("-p", "--passes", dest = "passes", type="int", default = 10,
        help = "specify how many passes to do training and validating, default is 10.")
    return parser.parse_args()

def is_file_type(filename, file_flags):
    for flag in file_flags:
        if flag.lower().strip() in filename.lower():
            return True
    return False

def preprocess(options):
    cfdna_flags = options.cfdna_flag.split(";")
    other_flags = options.other_flag.split(";")
    print("cfdna file flags:")
    print(cfdna_flags)
    print("other file flags:")
    print(other_flags)

    print("\nextracting features...")
    data = []
    label = []
    samples = []
    fq_files = get_arg_files()
    number = 0
    for fq in fq_files:
        if is_file_type(fq, cfdna_flags) == False and is_file_type(fq, other_flags) == False:
            continue
        number += 1
        print(str(number) + ": " + fq)

        extractor = FeatureExtractor(fq)
        extractor.extract()
        feature = extractor.feature()

        if feature == None:
            print("======== Warning: bad feature from:")
            print(fq)
            continue

        if is_file_type(fq, cfdna_flags):
            data.append(feature)
            label.append(1)
        elif is_file_type(fq, other_flags):
            data.append(feature)
            label.append(0)
        samples.append(fq)

    return data, label, samples

def random_separate(data, label, samples, training_set_percentage = 0.8):
    training_set = {"data":[], "label":[], "samples":[]}
    validation_set = {"data":[], "label":[], "samples":[]}
    total_num = len(data)
    training_num = int(round(total_num * training_set_percentage))
    if training_num == total_num:
        training_num -= 1
    if training_num < 2:
        training_num = 2

    # we should make sure the training set contains both positive and negative samples
    while( len(np.unique(training_set["label"])) <= 1 ):
        training_ids = random.sample([x for x in xrange(total_num)], training_num)
        training_set["data"] = []
        training_set["label"] = []
        training_set["samples"] = []
        validation_set["data"] = []
        validation_set["label"] = []
        validation_set["samples"] = []
        for i in xrange(total_num):
            if i in training_ids:
                training_set["data"].append(data[i])
                training_set["label"].append(label[i])
                training_set["samples"].append(samples[i])
            else:
                validation_set["data"].append(data[i])
                validation_set["label"].append(label[i])
                validation_set["samples"].append(samples[i])

    return training_set, validation_set

def train_svm(data, label, samples, options):
    print("\ntraining and validating using SVM for " + str(options.passes) + " times...")
    scores = 0
    for i in xrange(options.passes):
        training_set, validation_set = random_separate(data, label, samples)
        clf = svm.LinearSVC()
        clf.fit(np.array(training_set["data"]), np.array(training_set["label"]))
        score = clf.score(np.array(validation_set["data"]), np.array(validation_set["label"]))
        print("score: " + str(score))
        scores += score

    print("\naverage score: " + str(scores/options.passes))

def main():
    time1 = time.time()
    if sys.version_info.major >2:
        print('python3 is not supported yet, please use python2')
        sys.exit(1)

    (options, args) = parseCommand()

    data, label, samples = preprocess(options)

    if len(data) == 0:
        print("no enough training data, usage:\n\tpython training.py <fastq_files>\twildcard(*) is supported\n")
        sys.exit(1)
    elif len(np.unique(label)) < 2:
        if np.unique(label) == 0:
            print("no cfdna training data")
        else:
            print("no gdna training data")
        sys.exit(1)

    train_svm(data, label, samples, options)

    time2 = time.time()
    print('\nTime used: ' + str(time2-time1))

if __name__  == "__main__":
    main()