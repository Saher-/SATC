from nltk.corpus import movie_reviews
import TCS.Classifiers.Train as Train
from ..Classifiers import Classifier_NB
from ..Classifiers import Classifier_DT
from ..Classifiers import Classifier_ME
from ..Classifiers import Classifier_MV
from flask import Flask, render_template
import TCS.Classifiers.Sys_Params as Params

app = Flask(__name__)

nb_final = {}
me_final = {}
dt_final = {}
mv_final = {}


def data_run():
    # print "Preparing Data..."
    labels = movie_reviews.categories()
    labeled_words = [(l, movie_reviews.words(categories=[l])) for l in labels]
    high_info_words = set(Params.high_information_words(labeled_words))
    feat_det = lambda words: Params.bag_of_words_in_set(words, high_info_words)
    feats = Train.label_feat_from_corps(movie_reviews, feature_detector=feat_det)
    return Train.split_label_feats(feats)


def nb_test(test):
    nb = Classifier_NB.load_classifier()

    training, testing = data_run()

    nb_final[1] = nb.classify(Params.bag_of_words(test.lower()))
    nb_final[2] = round((Classifier_NB.Classifier_acc(nb, testing))*100, 2)
    # nb_final[2] = 00.00
    return nb_final


def dt_test(test):
    dt = Classifier_DT.load_classifier()

    training, testing = data_run()

    dt_final[1] = dt.classify(Params.bag_of_words(test))
    dt_final[2] = round((Classifier_DT.Classifier_acc(dt, testing))*100, 2)
    # dt_final[2] = 00.00
    return dt_final


def me_test(test):
    me = Classifier_ME.load_classifier()

    training, testing = data_run()

    me_final[1] = me.classify(Params.bag_of_words(test))
    me_final[2] = round((Classifier_DT.Classifier_acc(me, testing))*100, 2)
    # me_final[2] = 00.00
    return me_final


def mv_test(test):
    nb = Classifier_NB.load_classifier()
    dt = Classifier_DT.load_classifier()
    me = Classifier_ME.load_classifier()
    inst = Classifier_MV.MaxVoteClassifier(nb, dt, me)
    # inst.save_classifier(inst)
    mv = inst.load_classifier()

    training, testing = data_run()

    mv_final[1] = mv.classify(Params.bag_of_words(test))
    mv_final[2] = round((mv.Classifier_acc(mv, testing))*100, 2)
    # mv_final[2] = 00.00
    return mv_final


@app.route('/run_test')
def run_test(test, classifier, out):
    if classifier == 'NB':
        nb_test(test)
        if out == 'table':
            return render_template('single_test/single_test_output_nb_table.html', val_nb=nb_final, val_me=me_final, val_dt=dt_final, val_mv=mv_final)
        else:
            return render_template('single_test/single_test_output_nb_text.html', val_nb=nb_final, val_me=me_final, val_dt=dt_final, val_mv=mv_final)
    elif classifier == 'DT':
        dt_test(test)
        if out == 'table':
            return render_template('single_test/single_test_output_dt_table.html', val_nb=nb_final, val_me=me_final, val_dt=dt_final, val_mv=mv_final)
        else:
            return render_template('single_test/single_test_output_dt_text.html', val_nb=nb_final, val_me=me_final, val_dt=dt_final, val_mv=mv_final)
    elif classifier == 'ME':
        me_test(test)
        if out == 'table':
            return render_template('single_test/single_test_output_me_table.html', val_nb=nb_final, val_me=me_final, val_dt=dt_final, val_mv=mv_final)
        else:
            return render_template('single_test/single_test_output_me_text.html', val_nb=nb_final, val_me=me_final, val_dt=dt_final, val_mv=mv_final)
    elif classifier == 'MV':
        mv_test(test)
        if out == 'table':
            return render_template('single_test/single_test_output_mv_table.html', val_nb=nb_final, val_me=me_final, val_dt=dt_final, val_mv=mv_final)
        else:
            return render_template('single_test/single_test_output_mv_text.html', val_nb=nb_final, val_me=me_final, val_dt=dt_final, val_mv=mv_final)
    else:
        nb_test(test)
        me_test(test)
        dt_test(test)
        mv_test(test)
        if out == 'table':
            return render_template('single_test/single_test_output_all_table.html', val_nb=nb_final, val_me=me_final, val_dt=dt_final, val_mv=mv_final)
        else:
            return render_template('single_test/single_test_output_all_text.html', val_nb=nb_final, val_me=me_final, val_dt=dt_final, val_mv=mv_final)





