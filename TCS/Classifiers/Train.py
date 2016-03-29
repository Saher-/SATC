import re
import Sys_Params
import collections
import Classifier_NB
import Classifier_ME
import Classifier_DT
import Classifier_MV
from flask import render_template
from nltk.corpus import movie_reviews


op0 = re.compile('Op0', re.IGNORECASE)
op1 = re.compile('Op1', re.IGNORECASE)
op2 = re.compile('Op2', re.IGNORECASE)
op3 = re.compile('Op3', re.IGNORECASE)


def label_feat_from_corps(corp, feature_detector=Sys_Params.bag_of_words):
    """
    To create a list of labeled feature set [(featureset, label)]
    featureset -> is a Dict
    label -> the known class for the featureset (pos / neg)
    :param corp: any kind of corpus (movie reviews)
    :param feature_detector: a function to return a Dict (Default to bag_of_words)
    :return: a mapping set of {label: [featureset]}
    """
    label_feats = collections.defaultdict(list)
    for label in corp.categories():
        for fileid in corp.fileids(categories=[label]):
            feats = feature_detector(corp.words(fileids=[fileid]))
            label_feats[label].append(feats)
    return label_feats


def split_label_feats(lfeats, split=0.75):
    """
    To split that labeled mapped set to training & testing sets
    :param lfeats: mapping set returned from the label_feat_from_corps
    :param split: the ratio to split
    :return: two sets for training & testing
    """
    trainset = []
    testset = []
    for label, feats in lfeats.iteritems():
        cutoff = int(len(feats) * split)
        trainset.extend([(feat, label) for feat in feats[:cutoff]])
        testset.extend([(feat,label) for feat in feats[cutoff:]])
    return trainset, testset


def train_func(func1=Sys_Params.remove_punctuation,
               func2=Sys_Params.non_stop_words,
               func3=Sys_Params.do_pos,
               func4=Sys_Params.do_lmtize_pos,
               func5=Sys_Params.high_information_words, flag=2):

    tst = "This should be a GOOD TEST"
    if flag != 3:
        def func_final(tst):
            tst = func1(tst)
            tst = func2(tst)
            tst = func3(tst)
            ans = func4(tst)
            # ans = Sys_Params.bag_of_words(tst)
            return ans
        # final_func = Sys_Params.bag_of_words(func4(func3(func2(func1()))))
        print "Passing the function"
        labels = movie_reviews.categories()
        labeled_words = [(l, movie_reviews.words(categories=[l])) for l in labels]
        high_info_words = set(Sys_Params.high_information_words(labeled_words))
        feat_det = lambda words: Sys_Params.bag_of_words_in_set(func_final(words), high_info_words)
        # feat_det = lambda words: Sys_Params.bag_of_words_in_set(words, high_info_words)
        feats = label_feat_from_corps(movie_reviews, feature_detector=feat_det)
        # # print final_func
        # return final_func
    elif flag == 3:
        def func_final(tst):
            tst = func1(tst)
            tst = func2(tst)
            tst = func3(tst)
            tst = func4(tst)
            ans = Sys_Params.bag_of_words(tst)
            return ans
        # final_func = Sys_Params.bag_of_words(func4(func3(func2(func1()))))
        print "Passing the function"
        feats = label_feat_from_corps(movie_reviews, func_final)
        # feats = label_feat_from_corps(movie_reviews)
        # print final_func
        # return final_func

    training, testing = split_label_feats(feats)
    Classifier_NB.run(training)
    nb_Classifier = Classifier_NB.load_classifier()
    Classifier_DT.run(training)
    dt_Classifier = Classifier_DT.load_classifier()
    Classifier_ME.run(training)
    me_Classifier = Classifier_ME.load_classifier()
    inst = Classifier_MV.MaxVoteClassifier(nb_Classifier, dt_Classifier, me_Classifier)
    inst.save_classifier(inst)
    print "******DONE TRAINING******"
    return


def train_func_default():
    print "Passing the function"
    labels = movie_reviews.categories()
    labeled_words = [(l, movie_reviews.words(categories=[l])) for l in labels]
    high_info_words = set(Sys_Params.high_information_words(labeled_words))
    feat_det = lambda words: Sys_Params.bag_of_words_in_set(words, high_info_words)
    feats = label_feat_from_corps(movie_reviews, feature_detector=feat_det)
    training, testing = split_label_feats(feats)
    Classifier_NB.run(training)
    nb_Classifier = Classifier_NB.load_classifier()
    Classifier_DT.run(training)
    dt_Classifier = Classifier_DT.load_classifier()
    Classifier_ME.run(training)
    me_Classifier = Classifier_ME.load_classifier()
    inst = Classifier_MV.MaxVoteClassifier(nb_Classifier, dt_Classifier, me_Classifier)
    inst.save_classifier(inst)
    print "******DONE TRAINING******"
    return




def start_train(query):
    param = [0]*7

    if op0.findall(query):
        train_func_default()
        param[2] = 1
        param[5] = 1
        param[6] = 1

    elif op1.findall(query):
        train_func(Sys_Params.remove_punctuation,
                   Sys_Params.lower_case,
                   Sys_Params.non_stop_words,
                   Sys_Params.do_lmtize,
                   Sys_Params.high_information_words, 1)
        param[0] = 1
        param[1] = 1
        param[2] = 1
        param[4] = 1
        param[6] = 1

    elif op2.findall(query):
        train_func()
        param[0] = 1
        param[2] = 1
        param[3] = 1
        param[4] = 1
        param[6] = 1
    elif op3.findall(query):
        train_func(Sys_Params.remove_punctuation,
                   Sys_Params.non_stop_words,
                   Sys_Params.do_pos,
                   Sys_Params.do_lmtize_pos, '', 3)
        param[0] = 1
        param[2] = 1
        param[3] = 1
        param[4] = 1


    print param
    # return param
    return render_template('parameters/train_params.html', val=param)
    # return render_template('single_test/single_test.html')