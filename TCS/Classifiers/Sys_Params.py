import string
import collections
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from nltk.metrics import BigramAssocMeasures
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist, ConditionalFreqDist


#############
# IN POINT #
###########
def prepare(stat):
    """
    To prepare a normal string for pre-processing
    :param stat: Strin
    :return: list of Strings including the original string words
    """
    return stat.split()


############
# PHASE 1 #
##########
def remove_punctuation(stat):
    """
    To remove the Punctuation from the set
    :param stat: String of words "might include punctuation"
    :return: String of words excluding punctuation
    """
    print "************************"
    print "Removing Punctuation"
    # stat = prepare(stat)
    # if isinstance(stat, unicode):
    #     res = [str(word).translate(None, string.punctuation) for word in stat]
    # else:
    #     res = [word.translate(None, string.punctuation) for word in stat]
    res = [word.encode('utf-8').translate(None, string.punctuation) for word in stat]
    return ' '.join(res)


def lower_case(stat):
    """
    To Make all LOW CASE
    :param stat: String of words
    :return: String of LOWER CASE words
    """
    print "Lowering Cases"
    stat = prepare(stat)
    res = [word.lower() for word in stat]
    return ' '.join(res)


def non_stop_words(stat, sw='english'):
    """
    To make a list of FILTERED words to be used
    against all english stop words
    :param stat: String of words to be filtered
    :param sw:language of the stopwords
    :return: String of filtered words to be USED
    """
    print "Removing Stopwords"
    stat = prepare(stat)
    badwords = stopwords.words(sw)
    res = set(stat) - set(badwords)
    return ' '.join(res)


############
# PHASE 2 #
##########
def do_pos(stat, flag=1):
    """
    To do Part Of Speech Tagging POS on a sentence using POS_TAG
    :param stat: A NORMAL sentence string
    :param flag: To see if the output should be String or Tokens
    :return: Tokenizer of every word in the sentence and the POS Tag
    """
    print "POS Tagging"
    ans = pos_tag(word_tokenize(stat))
    res = [item for item in ans if item[1] == 'JJ']
    res += [item for item in ans if item[1] == 'JJR']
    res += [item for item in ans if item[1] == 'JJS']
    res += [item for item in ans if item[1] == 'RB']
    res += [item for item in ans if item[1] == 'RBR']
    res += [item for item in ans if item[1] == 'RBS']
    if flag == 1:
        return res
    elif flag == 0:
        temp = [item[0] for item in ans]
        return ' '.join(temp)


def do_lmtize_pos(stat):
    """
    To Make limmatization AFTER DOING POS first
    :param stat: Tokenizers of words & POS tags
    :return: limatized set of words
    """
    print "Lemmatizing"
    lmtzr = WordNetLemmatizer()
    res = ''
    for word, tag in stat:
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
        if not wntag:
            lemma = word
        else:
            lemma = lmtzr.lemmatize(word, wntag)
        res += lemma + ' '
    return res


def do_lmtize(stat):
    """
    To Make Limmatization (dogs -> dos / churches -> church / we -> we-be)
    WITHOUT doing POS first, (default None)
    :param stat: String of words
    :return: String of limmatized words
    """
    print "Lemmatizing"
    stat = prepare(stat)
    lmtzr = WordNetLemmatizer()
    res = [lmtzr.lemmatize(word) for word in stat]
    return ' '.join(res)


def bigram_words(stat, score_fn=BigramAssocMeasures.phi_sq, n=200):
    """
    To get a list of coupled words from a list of words
    :param stat: String of words
    :param score_fn: function to be used in creating the Bigram
    :param n: the number of the most significent bigrams needed
    :return: list of bigram words and pass it to bag_of_words to get a Dict
    """
    stat = prepare(stat)
    bigram_finder = BigramCollocationFinder.from_words(stat)
    bigrams = bigram_finder.nbest(score_fn, n)
    # filtered_bigram = Featx.bag_of_non_stop_words(bigrams).items()
    res = [i for i in stat]
    return bigrams + res


############
# PHASE 3 #
##########
def bag_of_words_in_set(words, goodwords):
    """
    To get a Dict of the most informative feature words
    :param words: set of words
    :param goodwords: set of high informative words
    :return: a combination of high informative words + words and pass it to bag_of_words
    """
    return bag_of_words(set(words) & set(goodwords), 1)


def high_information_words(labeled_words, score_fn=BigramAssocMeasures.chi_sq, min_score=5):
    """
    To eliminate low information feature words for set of words for EFFICIENCY
    :param labeled_words: list of 2 tuples [(label, words)]
                          label -> is a classification label (pos / neg)
                          words -> is a list of words that occur under that label
    :param score_fn: a scoring function to measure how informative that word is
    :param min_score: the minimum score for a word to be included as MOST INFORMATIVE WORD
    :return: a set of high informative words
    """
    print "Counting Word Frequencies"
    word_fq = FreqDist()
    labeled_word_fq = ConditionalFreqDist()

    for label, words in labeled_words:
        for word in words:
            word_fq[word] += 1
            labeled_word_fq[label][word] += 1
    n_xx = labeled_word_fq.N()
    high_info_words = set()

    for label in labeled_word_fq.conditions():
        n_xi = labeled_word_fq[label].N()
        word_scores = collections.defaultdict(int)

        for word, n_ii in labeled_word_fq[label].iteritems():
            n_ix = word_fq[word]
            score = score_fn(n_ii, (n_ix, n_xi), n_xx)
            word_scores[word] = score

        bestwords = [word for word, score in word_scores.iteritems() if score >= min_score]
        high_info_words |= set(bestwords)

    return high_info_words


##############
# OUT POINT #
############
def bag_of_words(words, flag=0):
    """
    to convert a list of words into a boolean Dict
    :param words: any String of words
    :return: Dict [(word, True)]
    """
    if flag == 0:
        stat = prepare(words)
        return dict([(word, True) for word in stat])
    elif flag == 1:
        return dict([(word, True) for word in words])


def bag_of_words_testing(words):
    """
    to convert a list of words into a boolean Dict
    :param words: any list of words
    :return: Dict [(word, True)]
    """
    return dict([(word, True) for word in words])


###############
# TEST BLOCK #
#############
# test = "The movie was SO GOOD, but this is not going to be as good !!!"
# print "Starting..."
# sample = prepare(test)
# print sample
# print "prepare DONE..."

# sample = remove_punctuation(test)
# print sample
# print "Punctuation DONE..."

# sample = lower_case(test)
# print sample
# print "Lower casing DONE..."
#
# sample = non_stop_words(test)
# print sample
# print "Stop words DONE..."
#
# tags = do_pos(test, 1)
# print sample
# res = [item for item in sample if item[1] == 'DT']
# res += [item for item in sample if item[1] == 'NN']
# res = [item[0] for item in res]
# print res
# print "POS DONE..."
#
# sample = do_lmtize_pos(tags)
# print sample
# print "Limmatization DONE..."
#
# sample = bigram_words(test)
# print sample
# print "Bigram DONE..."
#
# sample = bag_of_words(sample)
# print sample
# print "READY..."

# tst = "This ... should be a GOOD TEST, i am not sure about this one, i hope it is worth it !!!!!"
# tst = remove_punctuation(tst)
# print tst
# tst = non_stop_words(tst)
# print tst
# tst = do_pos(tst)
# print tst
# tst = do_lmtize_pos(tst)
# print tst
# tst = bag_of_words(tst)
# print tst

# print remove_punctuation(u'<foo>!')
# tst = u'<foo>!'
# tst2 = 'string'
# print tst.encode()
# print tst2.encode()