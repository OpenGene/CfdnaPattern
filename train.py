#!/usr/bin/env python
import sys, os
import fastq
from optparse import OptionParser
from multiprocessing import Process, Queue
import time
from util import *
from draw import *
from feature import *
import numpy as np
from sklearn import svm, neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
import random
import json
import pickle
from sklearn.model_selection import ShuffleSplit

def parseCommand():
    usage = "extract the features, and train the model, from the training set of fastq files. \n\npython training.py <fastq_files> [options] "
    version = "0.0.1"
    parser = OptionParser(usage = usage, version = version)
    parser.add_option("-m", "--model", dest = "model_file", default = "cfdna.model",
        help = "specify which file to store the built model.")
    parser.add_option("-a", "--algorithm", dest = "algorithm", default = "knn",
        help = "specify which algorithm to use for classfication, candidates are svm/knn/rbf/rf/gnb/benchmark, rbf means svm using rbf kernel, rf means random forest, gnb means Gaussian Naive Bayes, benchmark will try every algorithm and plot the score figure, default is knn.")
    parser.add_option("-c", "--cfdna_flag", dest = "cfdna_flag", default = "cfdna",
        help = "specify the filename flag of cfdna files, separated by semicolon. default is: cfdna")
    parser.add_option("-o", "--other_flag", dest = "other_flag", default = "gdna;ffpe",
        help = "specify the filename flag of other files, separated by semicolon. default is: gdna;ffpe")
    parser.add_option("-p", "--passes", dest = "passes", type="int", default = 100,
        help = "specify how many passes to do training and validating, default is 10.")
    parser.add_option("-n", "--no_cache_check", dest = "no_cache_check", action='store_true', default = False,
        help = "if the cache file exists, use it without checking the identity with input files")
    return parser.parse_args()

def is_file_type(filename, file_flags):
    for flag in file_flags:
        if flag.lower().strip() in filename.lower():
            return True
    return False

def preprocess(options):
    cfdna_flags = options.cfdna_flag.split(";")
    other_flags = options.other_flag.split(";")
    print("cfdna file flags (-c <cfdna_flags>): " + ";".join(cfdna_flags))
    print("other file flags (-o <other_flags>): " + ";".join(other_flags))

    data = []
    label = []
    samples = []
    fq_files = get_arg_files()

    # try to load from cache.json
    json_file_name = "cache.json"
    if os.path.exists(json_file_name) and os.access(json_file_name, os.R_OK):
        json_file = open(json_file_name, "r")
        json_loaded = json.loads(json_file.read())
        print("\nfound feature cache (cache.json), loading it now...")
        if options.no_cache_check or len(json_loaded["fq_files"]) == len(fq_files):
            data = json_loaded["data"]
            label = json_loaded["label"]
            samples = json_loaded["samples"]
            print("feature cache is valid, if you want to do feature extraction again, delete cache.json")
            return data, label, samples
        else:
            print("cache is invalid")

    # cannot load from cache.json, we compute it
    print("\nextracting features...")
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

        if has_adapter_sequenced(feature):
            continue

        if is_file_type(fq, cfdna_flags):
            data.append(feature)
            label.append(1)
        elif is_file_type(fq, other_flags):
            data.append(feature)
            label.append(0)
        else:
            continue
        samples.append(fq)

    if len(samples)<=2:
        return data, label, samples

    # save the data, label and samples to cache.json to speed up the training test
    try:
        json_file = open(json_file_name, "w")
    except Exception:
        return data, label, samples
    if os.access(json_file_name, os.W_OK):
        json_store = {}
        json_store["data"]=data
        json_store["label"]=label
        json_store["samples"]=samples
        json_store["fq_files"]=fq_files
        print("\nsave to cache.json")
        json_str = json.dumps(json_store)
        json_file.write(json_str)
        json_file.close()

    return data, label, samples

def random_separate(data, label, samples, training_set_percentage = 0.8):
    training_set = {"data":[], "label":[], "samples":[]}
    validation_set = {"data":[], "label":[], "samples":[]}
    total_num = len(data)
    
    ss = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)

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

def train(model, data, label, samples, options, benchmark = False):
    print("\ntraining and validating for " + str(options.passes) + " times...")
    total_score = 0
    scores = []
    wrong_files = []
    wrong_data = []
    total_num = len(data)
    
    ss = ShuffleSplit(n_splits=options.passes, test_size=0.2, random_state=0)
    passno = 0
    for train_index, test_index in ss.split(np.arange(total_num)):
        print(str(passno+1) + " / " + str(options.passes));
        passno += 1

        training_set = {}
        validation_set = {}
        training_set['data'] = [data[i] for i in train_index]
        validation_set['data'] = [data[i] for i in test_index]
        training_set['label'] = [label[i] for i in train_index]
        validation_set['label'] = [label[i] for i in test_index]
        training_set['samples'] = [samples[i] for i in train_index]
        validation_set['samples'] = [samples[i] for i in test_index]

        model.fit(np.array(training_set["data"]), np.array(training_set["label"]))
        # get scores
        score = model.score(np.array(validation_set["data"]), np.array(validation_set["label"]))
        total_score += score
        scores.append(score)

        # predict
        arr = np.array(validation_set["data"])
        for v in xrange(len(validation_set["data"])):
            result = model.predict(arr[v:v+1])
            if result[0] != validation_set["label"][v]:
                #print("Truth: " + str(validation_set["label"][v]) + ", predicted: " + str(result[0]) + ": " + validation_set["samples"][v])
                if validation_set["samples"][v] not in wrong_files:
                    wrong_files.append(validation_set["samples"][v])
                    wrong_data.append(validation_set["data"][v])
    if not benchmark:
        print("scores of all " + str(options.passes) + " passes:")
        print(scores)
        print("\naverage score: " + str(total_score/options.passes))
        print("\n" + str(len(wrong_files)) + " files with at least 1 wrong prediction:")
        print(" ".join(wrong_files))

        print("\nplotting figures for files with wrong predictions...")
        plot_data_list(wrong_files, wrong_data, "train_fig")

        save_model(model, options)
    return sorted(scores, reverse=True)

def save_model(model, options):
    print("\nsave model to: " + options.model_file)
    try:
        f = open(options.model_file, "wb")
        pickle.dump(model, f, True)
    except Exception:
        print("failed to write file")

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

    if options.algorithm.lower() == "svm":
        model = svm.LinearSVC()
        train(model, data, label, samples, options)
    elif options.algorithm.lower() == "knn":
        model = neighbors.KNeighborsClassifier(leaf_size=100)
        train(model, data, label, samples, options)
    elif options.algorithm.lower() == "rf":
        model = RandomForestClassifier(n_estimators=20)
        train(model, data, label, samples, options)
    elif options.algorithm.lower() == "rbf":
        model = svm.SVC(kernel='rbf')
        train(model, data, label, samples, options)
    elif options.algorithm.lower() == "gnb":
        model = GaussianNB()
        train(model, data, label, samples, options)
    elif options.algorithm.lower() == "benchmark":
        print("\nstarting benchmark...")
        names = ["KNN", "Random Forest","SVM Linear Kernel", "Gaussian Naive Bayes", "SVM RBF Kernel"]
        models = [neighbors.KNeighborsClassifier(leaf_size=100), RandomForestClassifier(n_estimators=20), svm.LinearSVC(), GaussianNB(), svm.SVC(kernel='rbf')]
        scores_arr = []
        for m in xrange(len(models)):
            print("\nbenchmark with: " + names[m])
            scores_arr.append(train(models[m], data, label, samples, options, True))
        print("\nploting benchmark result...")
        plot_benchmark(scores_arr, names, "benchmark.png")
    else:
        print("algorithm " + options.algorithm + " is not supported, please use svm/knn")

    time2 = time.time()
    print('\nTime used: ' + str(time2-time1))

if __name__  == "__main__":
    main()
