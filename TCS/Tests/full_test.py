import TCS.Classifiers.Train as Train
from nltk.corpus import movie_reviews
from ..Classifiers import Classifier_NB
from ..Classifiers import Classifier_ME
from ..Classifiers import Classifier_DT
from ..Classifiers import Classifier_MV
from flask import Flask, render_template
import TCS.Classifiers.Sys_Params as Params


app = Flask(__name__)

nb_final = {}
me_final = {}
dt_final = {}
mv_final = {}


def data_run():
    print "Preparing Data..."
    labels = movie_reviews.categories()
    labeled_words = [(l, movie_reviews.words(categories=[l])) for l in labels]
    high_info_words = set(Params.high_information_words(labeled_words))
    feat_det = lambda words: Params.bag_of_words_in_set(words, high_info_words)
    feats = Train.label_feat_from_corps(movie_reviews, feature_detector=feat_det)
    return Train.split_label_feats(feats)


def nb_test():
    # Classifier_NB.run(training)

    training, testing = data_run()

    nb = Classifier_NB.load_classifier()

    np_p, np_r = Classifier_NB.precision_recall(nb, testing)

    print "Results are out!"

    nb_final[1] = round((Classifier_NB.Classifier_acc(nb, testing))*100, 2)

    nb_final[2] = round((np_p['pos'])*100, 2)
    nb_final[3] = round((np_p['neg'])*100, 2)

    nb_final[4] = round((np_r['pos'])*100, 2)
    nb_final[5] = round((np_r['neg'])*100, 2)

    return nb_final


def me_test():
    # Classifier_NB.run(training)

    training, testing = data_run()

    print "Loading Classifier..."
    me = Classifier_ME.load_classifier()

    me_p, me_r = Classifier_ME.precision_recall(me, testing)

    print "Results are out!"

    me_final[1] = round((Classifier_ME.Classifier_acc(me, testing))*100, 2)

    me_final[2] = round((me_p['pos'])*100, 2)
    me_final[3] = round((me_p['neg'])*100, 2)

    me_final[4] = round((me_r['pos'])*100, 2)
    me_final[5] = round((me_r['neg'])*100, 2)

    return me_final


def dt_test():
    # Classifier_NB.run(training)

    training, testing = data_run()

    dt = Classifier_DT.load_classifier()

    dt_p, dt_r = Classifier_DT.precision_recall(dt, testing)

    print "Results are out!"

    dt_final[1] = round((Classifier_DT.Classifier_acc(dt, testing))*100, 2)

    dt_final[2] = round((dt_p['pos'])*100, 2)
    dt_final[3] = round((dt_p['neg'])*100, 2)

    dt_final[4] = round((dt_r['pos'])*100, 2)
    dt_final[5] = round((dt_r['neg'])*100, 2)

    return dt_final


def mv_test():
    nb = Classifier_NB.load_classifier()
    dt = Classifier_DT.load_classifier()
    me = Classifier_ME.load_classifier()
    inst = Classifier_MV.MaxVoteClassifier(nb, dt, me)
    # inst.save_classifier(inst)
    # mv = inst.load_classifier()

    training, testing = data_run()

    mv_p, mv_r = inst.precision_recall(inst, testing)

    print "Results are out!"

    mv_final[1] = round(inst.Classifier_acc(inst, testing)*100, 2)

    mv_final[2] = round((mv_p['pos'])*100, 2)
    mv_final[3] = round((mv_p['neg'])*100, 2)

    mv_final[4] = round((mv_r['pos'])*100, 2)
    mv_final[5] = round((mv_r['neg'])*100, 2)

    return mv_final


@app.route('/run_test')
def run_test():
    nb_test()
    me_test()
    dt_test()
    mv_test()
    return render_template('full_test/full_test.html', val_nb=nb_final, val_me=me_final, val_dt=dt_final, val_mv=mv_final)


# return render_template('full_test.html', nb_func1=acc_test, nb_func2=pp_test, nb_func3=pn_test, nb_func4=rp_test, nb_func5=rn_test)
