import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from typing import List
from vypryamitel.dataset import SDataset
from .data_downloader import *
from .langutils import can_be_switched, switch_layout, symb_eng_str, symb_rus_str, is_word
# from vypryamitel.query_processing_logic.dictionary import Dictionary
from .utils import get_words_from_dict

digits_count = lambda s: sum(c.isdigit() for c in s)


class KeyboardSwitcher():
    def __init__(self, dictionary, vertical: str):
        self.dictionary = dictionary
        self.vertical = vertical

    def get_features(self, word, bigrams, trigrams):
        """
        :param word: word
        :param symbols_rus: list of symbols of russian alphabet
        :param symbols_eng: list of symbols of english alphabet
        :param bigrams:  list of bigrams 
        :param trigrams: list of trigrams
        :return: list of features of current word by list of features
        """
        features = [digits_count(word)]
        mut_word = switch_layout(word)
        features.extend([word.count(symb) for symb in symb_eng_str + symb_rus_str])
        features.extend([word.count(ngr) - mut_word.count(ngr) for ngr in bigrams + trigrams])
        return features

    def matrix_constructing(self):
        words = set()
        for dataset in [self.dictionary.loader.get_site_dataset_by_config(site, feeds=True) for site in
                        self.dictionary.loader.get_vertical_sites(self.vertical)]:
            if dataset is not None:
                words.update(dataset.get_all_words_from_feed())
        feed = pd.DataFrame(data=list(words), columns=['text'])
        feed['text'] = feed.text.str.lower()
        feed = feed[~feed.text.str.isdigit()].drop_duplicates()
        feed_mut = pd.DataFrame(feed.text.apply(switch_layout))

        feed['target'] = 0
        feed_mut['target'] = 1
        ad_words = pd.DataFrame(data=list(
            self.dictionary.loader.additional_layout_dictionary | self.dictionary.loader.get_vertical_brands_set_words(
                self.vertical)), columns=['text'])
        ad_words['text'] = ad_words['text'].str.lower()
        ad_words = ad_words.drop_duplicates()

        ad_words_mut = pd.DataFrame(ad_words['text'].apply(switch_layout))
        ad_words['target'] = 0
        ad_words_mut['target'] = 1

        words = shuffle(pd.concat([feed, feed_mut])).head(10000)
        words = shuffle(pd.concat([words, ad_words, ad_words_mut]))
        words = words.reset_index(drop=True)
        trigrams_all = Counter()
        for index in words.index:
            word = words['text'][index]
            for i in range(len(word) - 2):
                t = word[i:i + 3]
                if '\n' not in t and not digits_count(t):
                    trigrams_all[t] += 1

        bigrams_all = Counter()
        for index in words.index:
            word = words['text'][index]
            for i in range(len(word) - 1):
                b = word[i:i + 2]
                if type(b) == str and '\n' not in b and not digits_count(b):
                    bigrams_all[b] += 1

        bigrams = sorted([b for b, _ in bigrams_all.most_common()[:1000]])
        trigrams = sorted([t for t, _ in trigrams_all.most_common()[:1000]])

        X = [self.get_features(word, bigrams, trigrams) for word in words.text]
        y = words['target']
        return X, y, bigrams, trigrams

    def model_creating(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        clf = LogisticRegression()
        model = clf.fit(X=X_train, y=y_train)

        predict = model.predict_proba(X_test)
        predict = [np.argmax(a) for a in predict]
        scorers = {'accuracy_score': accuracy_score(y_test, predict),
                   'f1_score': f1_score(y_test, predict),
                   'precision_score': precision_score(y_test, predict),
                   'roc_auc_score': roc_auc_score(y_test, predict)}

        return model, scorers

    def keyboard_switcher(self, search, model, bigrams, trigrams, probability_for_switch=0.5):
        words = search.split(' ')
        probas = model.predict_proba(np.array([self.get_features(word, bigrams, trigrams)
                                               for word in words]))[:, 1]

        new_search = ' '.join([word if len(word) < 2 \
                                       or proba < probability_for_switch
                                       or not can_be_switched(word) \
                                   else switch_layout(word).replace('/', 'ю')
                               for word, proba in zip(words, probas)])
        return {'response': new_search}
