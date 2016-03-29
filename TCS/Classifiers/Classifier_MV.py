import nltk
import pickle
import itertools
import collections
from nltk.classify import ClassifierI
from nltk.probability import FreqDist



class MaxVoteClassifier(ClassifierI):
    """
    a class that extends ClassifierI to combine the classifiers
    """
    def __init__(self, *classifier):
        """
        Constructor
        :param classifier: a number of classifiers needs to be combined
        :return: a combined classifier object that works as anyother classifier
        """
        self._classifier = classifier
        self._labels = sorted(set(itertools.chain(*[c.labels() for c in classifier])))

    def labels(self):
        """
        :return: classifier labels
        """
        return self._labels

    def classify(self, featureset):
        """
        :param featureset: set of feature needs to be classified
        :return: return the label of the given feature set, after voting its classifiers
        """
        counts = FreqDist()

        for classifier in self._classifier:
            counts[classifier.classify(featureset)] =+ 1

        return counts.max()

    def Classifier_acc(self, C, test_set):
        return nltk.classify.accuracy(C, test_set)

    def precision_recall(self, C, test_set):
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

    def save_classifier(self, C):
        """
        To save the trained Classifier
        so that it doesn't need to be re-trained every time
        :param C: trained classifier
        :return: outputs a traind classifier pickle
        """
        f = open('Classifiers/mv.pickle', 'wb')
        pickle.dump(C, f)
        f.close()
    
    def load_classifier(self):
        """
        To load a trained classifier ready to be used
        :return: a trained instance of a classifier
        """
        print "Loading MV Classifier..."
        f = open('Classifiers/mv.pickle', 'rb')
        C = pickle.load(f)
        f.close
        return C





# labels = movie_reviews.categories()
# labeled_words = [(l, movie_reviews.words(categories=[l])) for l in labels]
# high_info_words = set(Featx.high_information_words(labeled_words))
# feat_det = lambda words: Featx.bag_of_words_in_set(words, high_info_words)
# feats = Classifier_NB.label_feat_from_corps(movie_reviews, feature_detector=feat_det)
# feats = Classifier_NB.label_feat_from_corps(movie_reviews)
# training, testing = Classifier_NB.split_label_feats(feats)
#
# nb = Classifier_NB.load_classifier()
# dt = Classifier_DT.load_classifier()
# me = Classifier_ME.load_classifier()
#
# mv = MaxVoteClassifier(nb, dt, me)
# print mv.labels()
#
# print "Running Classifier"
# # run()
# print "Classifier trained"
# print "Loading Classifier"
# test_sent = "This movie is too bad to watch"
# print test_sent
# print mv.classify(Featx.bag_of_words(test_sent))
# print C.show_most_informative_features(5)
# print "Classifier Accuracy: " + str(mv.Classifier_acc(mv, testing))
# mv_p, mv_r = Classifier_NB.precision_recall(mv, testing)
# print "Precision Positive" + str(mv_p['pos'])
# print "Precision Negative" + str(mv_p['neg'])
# print "Recall Positive" + str(mv_r['pos'])
# print "Recall Negative" + str(mv_r['neg'])

