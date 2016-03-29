import nltk
import pickle
import collections
from nltk.classify import NaiveBayesClassifier


def Classifier_acc(C, test_set):
    """
    to determine the accuracy of a classifier on a given test set
    :param C: Classifier
    :param test_set: test set
    :return: Accuracy percentage
    """
    return nltk.classify.accuracy(C, test_set)


def precision_recall(C, test_set):
    """
    :param C: trained classifier
    :param test_set: testing set
    :return: two Dict 1st holds the precision for each label
                      2nd holds the recall for each label
    """
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(test_set):
            refsets[label].add(i)
            observed = C.classify(feats)
            testsets[observed].add(i)

    precisions = {}
    recalls = {}

    for label in C.labels():
            precisions[label] = nltk.precision(refsets[label], testsets[label])
            recalls[label] = nltk.recall(refsets[label], testsets[label])
    return precisions, recalls


def run(training):
    """
    To create and train a NaiveBayesClassifier
    :return: a trained Classifier
    """
    print "Training NB Classifier..."
    # feats = label_feat_from_corps(movie_reviews)
    # training, testing = split_label_feats(feats)

    nb_classifier = NaiveBayesClassifier.train(training)
    print "NB Classifier trained..."
    return save_classifier(nb_classifier)


def save_classifier(C):
    """
    To save the trained Classifier
    so that it doesn't need to be re-trained every time
    :param C: trained classifier
    :return: outputs a traind classifier pickle
    """
    f = open('Classifiers/nb.pickle', 'wb')
    pickle.dump(C, f)
    f.close()


def load_classifier():
    """
    To load a trained classifier ready to be used
    :return: a trained instance of a classifier
    """
    print "Loading NB Classifier..."
    f = open('Classifiers/nb.pickle', 'rb')
    C = pickle.load(f)
    f.close()
    return C


# labels = movie_reviews.categories()
# labeled_words = [(l, movie_reviews.words(categories=[l])) for l in labels]
# high_info_words = set(Featx.high_information_words(labeled_words))
# feat_det = lambda words: Featx.bag_of_words_in_set_nsw(words, high_info_words)
# feats = label_feat_from_corps(movie_reviews, feature_detector=feat_det)
# feats = label_feat_from_corps(movie_reviews)
# training, testing = split_label_feats(feats)
# print "Running Classifier"
# run()
# print "Classifier trained"
# print "Loading Classifier"
# C = load_classifier()
# test_sent = "This movie the worst movie ever"
# print test_sent
# print C.classify(Featx.bag_of_words(test_sent))
# print C.show_most_informative_features(5)
# print "Classifier Accuracy: " + str(Classifier_acc(C, testing))
# np_p, np_r = precision_recall(C, testing)
# print "Precision Positive: " + str(np_p['pos'])
# print "Precision Negative: " + str(np_p['neg'])
# print "Recall Positive: " + str(np_r['pos'])
# print "Recall Negative: " + str(np_r['neg'])