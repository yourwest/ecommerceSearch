import xgboost as xgb
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import editdistance

from common.utils import LoggerMixin
from .dictionary import Dictionary
from .loader import Loader
from .query import Query
from .serializable import Serializable
from vypryamitel.langutils import switch_layout, is_word, split_set_of_words
from vypryamitel.weighted_levenshtein import my_weighted_levenshtein

from typing import DefaultDict, Set
from collections import defaultdict
import numpy as np
import pandas as pd


class TyposClassifier(LoggerMixin):
    def __init__(self, frequencies_search, frequencies_feed, bags, feed, dictionary_with_splitting):
        self.frequencies_search = frequencies_search
        self.frequencies_feed = frequencies_feed
        self.bags = bags
        self.feed = feed
        self.dictionary_with_splitting = dictionary_with_splitting

    def _get_features(self, fasttext_model, sentence: list, word: str, candidate: str, fuzzy_result):
        stop_words = set(stopwords.words('english') + stopwords.words('russian'))
        fuzzy_res_cand = [(res, w) for res, w in fuzzy_result if w == candidate]
        if len(fuzzy_res_cand) > 0:
            fuzzy_score = fuzzy_res_cand[0][0] / fuzzy_result[0][0]
        else:
            fuzzy_score = 0
        sentence_bag = set()
        for w in sentence:
            if w in self.bags:
                sentence_bag.update(self.bags[w])
        dist_to_word = cosine(np.array(fasttext_model[word]), np.array(fasttext_model[candidate]))
        dist_to_sentence = np.array(
            [cosine(np.array(fasttext_model[w]), np.array(fasttext_model[candidate])) for w in
             sentence if (len(w) > 1 and w not in stop_words and w != word)])
        if len(dist_to_sentence) == 0:
            dist_to_sentence = dist_to_word
        else:
            dist_to_sentence = dist_to_sentence.mean()
        return [my_weighted_levenshtein(word, candidate),
                editdistance.eval(word, candidate),
                fuzzy_score,
                dist_to_word,
                dist_to_sentence,
                self.frequencies_search.get(word, 0),
                self.frequencies_search.get(candidate, 0),
                self.frequencies_feed.get(candidate, 0),
                (candidate in sentence_bag) or (len(sentence_bag) == 0),
                candidate in self.feed,
                len(sentence)
                ]

    def _get_matrix(self, data, fasttext_model, fuzzy_set):
        matrix = []
        target = []
        self.info('getting matrix of features, ' + str(len(data)) + ' rows')
        cur_pair = data.search[0], data.word[0]
        cur_result = fuzzy_set.get(data.word[0])
        for index in data.index:
            pair = data.search[index], data.word[index]
            if pair != cur_pair:
                cur_pair = data.search[index], data.word[index]
                cur_result = fuzzy_set.get(data.word[index])
            try:
                row = self._get_features(fasttext_model,
                                         data.search[index].split(' '), data.word[index],
                                         data.candidat[index], cur_result)
                matrix.append(row)
                target.append(data.answer[index])
            except:
                self.critical(
                    "Unable to process line %s %s %s" % (data.search[index], data.word[index], data.candidat[index]))
        return np.array(matrix), np.array(target)

    def create_model(self, fasttext_model, fuzzy_set, data: pd.DataFrame, number_negative: int):
        data = data.fillna('')
        self.candidates = pd.DataFrame(data=[])
        self.info(str(len(data)) + ' positive samples for train')
        for _, row in data.iterrows():
            sentence, word, r_candidate = row["sentence"], row['word'], row["right candidat"]
            if r_candidate != '' and r_candidate in self.dictionary_with_splitting:
                self.candidates = self.candidates.append([[sentence, word, r_candidate, 1]])
            else:
                if np.random.random() < 0.95:
                    continue
            cands = [c for _, c in fuzzy_set.get(word)][:100]
            if r_candidate in cands:
                cands.remove(r_candidate)
            for cand in cands:
                self.candidates = self.candidates.append([[sentence, word, cand, 0]])
        self.candidates.columns = ['search', 'word', 'candidat', 'answer']
        self.candidates = self.candidates.sort_values(['search', 'word']).reset_index()[
            ['search', 'word', 'candidat', 'answer']]
        matrix, target = self._get_matrix(self.candidates, fasttext_model, fuzzy_set)
        mt = pd.DataFrame(matrix)
        mt['target'] = target
        mt_pos = mt[mt['target'] == 1]
        mt_neg = mt[mt['target'] == 0]
        mt_neg = shuffle(mt_neg).head(len(mt_pos) * number_negative)
        mt = pd.concat([mt_pos, mt_neg])
        target_x = np.array(mt['target'])
        del mt['target']
        matrix_x = np.array(mt)
        X_train, X_test, y_train, y_test = train_test_split(matrix_x, target_x, test_size=0.05)
        clf = xgb.XGBClassifier()
        self.typos_classifier_model = clf.fit(X=X_train, y=y_train, verbose=True)
        predict = self.typos_classifier_model.predict_proba(X_test)
        predict = [np.argmax(a) for a in predict]
        self.typos_classifier_scorers = {'accuracy_score': accuracy_score(y_test, predict),
                                         'f1_score': f1_score(y_test, predict),
                                         'precision_score': precision_score(y_test, predict),
                                         'roc_auc_score': roc_auc_score(y_test, predict)}
        return self.typos_classifier_model, self.typos_classifier_scorers

    def test(self, number_negatives: list, high_thresholds: list, low_thresholds: list, contrasts: list):
        """
        This method is for validation treshholds for typos classifier - number negative samples for model fitting
        and treshholds for replace desision. The method returns dataframe  with 8 columns 
        ['number_negative', 'high_threshold', 'low_threshold', 'contrast', 'true_replace', 'false_replace', 
        'true_miss_replace', 'false_miss_replace']
        These parameters can be optimized by some loss function, and can be used in next intialisation of vypryamitel.
        It should be used after firs using self.model_creating
        """
        self.warning("This method is deprecated. Please refactor it and move in separate file in test/")
        result = pd.DataFrame()
        for number_negative in number_negatives:
            # ToDo: this functionality is broken because TyposClassifier no longer has .matrix / .target
            # (for sake of memory conservation)
            mt = pd.DataFrame(self.matrix)
            mt['target'] = self.target
            mt_pos = mt[mt['target'] == 1]
            mt_neg = mt[mt['target'] == 0]
            mt_neg = shuffle(mt_neg).head(len(mt_pos) * number_negative)
            mt = pd.concat([mt_pos, mt_neg])
            target_x = np.array(mt['target'])
            del mt['target']
            matrix_x = np.array(mt)
            clf = xgb.XGBClassifier()
            model = clf.fit(X=matrix_x, y=target_x, verbose=True)
            test = pd.DataFrame(self.matrix)
            test[['search', 'word', 'candidat', 'answer']] = self.candidates[['search', 'word', 'candidat', 'answer']]
            test['model'] = [p for _, p in model.predict_proba(np.array(test[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))]
            test = test[['search', 'word', 'candidat', 'answer', 'model']]
            test = test.sort(['model'], ascending=False)
            groups = test.groupby(['search', 'word'])
            for high_treshhold in high_thresholds:
                for low_treshhold in low_thresholds:
                    for contrast in contrasts:
                        true_replace = 0
                        false_replace = 0
                        true_miss_replace = 0
                        false_miss_replace = 0
                        for g in groups:
                            df = pd.DataFrame(g[1]).reset_index()
                            if len(df) > 1:
                                best_prob = df['model'][0]
                                next_prob = df.head(2)[1:]['model'].mean()
                                if len(df[df['answer'] == 1]) > 0:
                                    idx_best_prob = df.index[df['model'] == best_prob].tolist()[0]
                                    idx_true_answer = df.index[df['answer'] == 1].tolist()[0]
                                    if idx_best_prob == idx_true_answer:
                                        if best_prob >= high_treshhold or (
                                                        best_prob >= low_treshhold and best_prob * contrast > next_prob):
                                            true_replace += 1
                                        else:
                                            false_miss_replace += 1
                                    else:
                                        if best_prob >= high_treshhold or (
                                                        best_prob >= low_treshhold and best_prob * contrast > next_prob):
                                            false_replace += 1
                                        else:
                                            false_miss_replace += 1
                                else:
                                    if best_prob >= high_treshhold or (
                                                    best_prob >= low_treshhold and best_prob * contrast > next_prob):
                                        false_replace += 1
                                    else:
                                        true_miss_replace += 1
                        result = result.append([[number_negative, high_treshhold, low_treshhold, contrast, true_replace,
                                                 false_replace, true_miss_replace, false_miss_replace]])
        result.columns = ['number_negative', 'high_treshhold', 'low_treshhold', 'contrast', 'true_replace',
                          'false_replace', 'true_miss_replace', 'false_miss_replace']
        return result


class TypoFixer(Serializable):
    def __init__(self, loader: Loader):
        super().__init__(loader)
        self.loader = loader
        self.typos_classifier = dict()
        self.classifier_params = {'number_negative': 95,
                                  'high_treshhold': 0.7,
                                  'low_treshhold': 0.07,
                                  'contrast': 0.15}

    def get_bags_vertical(self, vertical: str) -> DefaultDict[str, set]:
        ans = defaultdict(lambda: set())
        for site in self.loader.get_vertical_sites(vertical=vertical):
            bags = self.loader.get_site_dataset_by_config(site, feeds=True).get_bags_of_words()
            for key in bags:
                ans[key].update(bags[key])
        return ans

    def update(self, verticals, dictionary: Dictionary):
        for vertical in verticals:
            fasttext_model = self.loader.get_fasttext_model(vertical)
            self.typos_classifier[vertical] = TyposClassifier(
                bags=self.get_bags_vertical(vertical=vertical),
                feed=dictionary.valid_words_for_vertical.get(vertical, set())
                     | self.loader.get_vertical_brands_set_words(vertical),
                dictionary_with_splitting=split_set_of_words(
                    dictionary.loader.general_dictionary | self.loader.get_vertical_dictionary(
                        vertical=vertical) - self.loader.blacklist),
                frequencies_feed=dictionary.frequencies_feed_vertical[vertical],
                frequencies_search=dictionary.frequencies_search_vertical[vertical]
            )
            self.typos_classifier[vertical].create_model(
                fasttext_model=fasttext_model,
                fuzzy_set=dictionary.fuzzy_sets[vertical],
                data=self.loader.get_vertical_typo_train_dataset(vertical),
                number_negative=self.classifier_params['number_negative'])

    def complicated_check_word(self, fasttext_model,
                               vertical: str, word: str, query: list,
                               high_threshold: float, low_threshold: float, contrast: float,
                               dictionary: Dictionary):
        if is_word(word) or is_word(switch_layout(word)):
            result = dictionary.fuzzy_sets[vertical].get(word)  # pairs of scores and candidates
            hash100, cand100 = tuple(zip(*result[:100]))  # top100 scores and candidates
            class_scores = self.typos_classifier[vertical].typos_classifier_model.predict_proba(
                # scores of classifier for candidates
                np.array([self.typos_classifier[vertical]._get_features(fasttext_model, query, word, c, result[:100])
                          for c in cand100]))
            k = min(len(cand100), 2)
            # print(pd.DataFrame([cand100, class_scores[:, 0], class_scores[:, 1]]).transpose().sort([1]).head())
            cand_idx = np.argpartition(-class_scores[:, 1], np.arange(k))[np.arange(k)]  # <- ids of the best candidates
            # smart threshold: if it sees that classifier can destinguish first candidate from nearest 5
            # it will return this candidate despite his probability being less than 0.4 (still it needs to be more that 0.05)
            if class_scores[cand_idx[0], 1] > high_threshold or (
                            class_scores[cand_idx[0], 1] > low_threshold and class_scores[
                        cand_idx[1:], 1].mean() < contrast * class_scores[cand_idx[0], 1]):
                word, w_conf = cand100[cand_idx[0]], class_scores[cand_idx[0], 1]
            else:
                w_conf = 0
        else:
            w_conf = 0
        return word, w_conf

    def fix_typos(self, query: Query, dictionary: Dictionary, fasttext_model) -> Query:
        """ Note: function updates tokens and confidence simultaneously. """
        for i in range(len(query.tokens)):
            if not query.confidence_of_tokens[i]:
                query.tokens[i], query.confidence_of_tokens[i] = \
                    self.complicated_check_word(vertical=query.site.vertical,
                                                word=query.tokens[i],
                                                query=query.tokens,
                                                high_threshold=self.classifier_params['high_treshhold'],
                                                low_threshold=self.classifier_params['low_treshhold'],
                                                contrast=self.classifier_params['contrast'],
                                                dictionary=dictionary,
                                                fasttext_model=fasttext_model
                                                )
        return query
