from TCS.Tests import full_test
from TCS.Classifiers import Train
from TCS.Tests import single_test
from flask import Flask, render_template, request

app = Flask(__name__)
app.secret_key = 'many random bytes'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/parameters', methods=['GET'])
def parameters():
    if request.query_string:
        return Train.start_train(request.query_string)
        # return render_template('parameters/parameters.html')
    else:
        return render_template('parameters/parameters.html')


@app.route('/single_test', methods=['GET', 'POST'])
def singletest():
    if request.method == 'POST':
        name = request.form['test-name']
        classifier = request.form['test-classifier']
        output = request.form['output']
        return single_test.run_test(name, classifier, output)
    else:
        return render_template('single_test/single_test.html')


@app.route('/full_test')
def fulltest():
    # return render_template('layout.html')
    # return render_template('fulltest.html')
    # return NB_Test.nb_test(), ME_Test.me_test()
    return full_test.run_test()


# @app.route('/train')
# def train():
#     return Train.start_train()


@app.route('/documentation')
def documentation():
    return render_template('documentation.html')
#
#
# @app.route('/fullTest')
# def fullTest():
#     # return render_template('fullTest.html', myfunction=test_func)
#     return test.fullTest()
#
#
# @app.route('/run')
# def run():
#     choice = "Tom Cruise"
#     nb = Classifier_NB.load_classifier()
#     dt = Classifier_DT.load_classifier()
#     me = Classifier_ME.load_classifier()
#     mv = Classifier_MV.MaxVoteClassifier(nb, dt, me)
#
#     test = Featx.bag_of_non_stop_words(choice.split())
#     msg = mv.classify(test)
#     return render_template('run.html', var1=msg)

if __name__ == '__main__':
    app.run(debug=True)
