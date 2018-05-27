from typing import List, Tuple, Set

from ctrie import CTrie
from ctypos import CTyposModel

from common.utils import LoggerMixin
from language_model.ngram_storage import NGramStorage
from language_model.language_model import LanguageModel

from .serializable import Serializable
from .dictionary import Dictionary
from .query import Query
from .loader import Loader
from ..langutils import is_word, switch_layout

import numpy as np
import re
from collections import Counter, defaultdict
from tqdm import tqdm
from nltk.corpus import stopwords
import pymorphy2
import enchant


class SpellcheckerParameters:
    def __init__(self, beam_size=2, threshold=-13.0, theta=0.5, gamma=1.5):
        self.beam_size = beam_size
        self.threshold = threshold
        self.theta = theta
        self.gamma = gamma


class Spellchecker(Serializable, LoggerMixin):
    def __init__(self, loader: Loader, default_parameters=SpellcheckerParameters(), vertical_parameters={}):
        super().__init__(loader)
        self.loader = loader
        self.trie = CTrie()
        self.typos_model = CTyposModel()

        self.default_parameters = default_parameters
        self.vertical_parameters = {
            'childhood': SpellcheckerParameters(beam_size=3, gamma=1.3, theta=0.5, threshold=-12.2),
            'food': SpellcheckerParameters(beam_size=2, gamma=1.1, theta=0.6, threshold=-10.0),
            'electronics': SpellcheckerParameters(beam_size=2, gamma=1.7, theta=0.7, threshold=-12.8)
        }
        self.vertical_parameters.update(vertical_parameters)

    def update(self, verticals, dictionary: Dictionary, chunk_size=50000000):
        print("Building Spellchecker")

        typos = []

        self.trie = CTrie()
        self.typos_model = CTyposModel()

        sentence_generator = dictionary._sentence_generator(verticals, True)
        update_finished = False

        while not update_finished:
            update_finished = True
            word_count = 0

            print("... Finding contexts")
            word_counter = Counter()
            context_count = Counter()
            context_words = defaultdict(Counter)
            word_contexts = defaultdict(set)
            for sentence in tqdm(sentence_generator):
                for i in range(len(sentence)):
                    word = sentence[i]
                    left_context = ''
                    right_context = ''
                    if i > 0:
                        left_context = sentence[i - 1]
                    if i + 1 < len(sentence):
                        right_context = sentence[i + 1]
                    word_counter[word] += 1
                    context_count[(left_context, right_context)] += 1
                    context_words[(left_context, right_context)][word] += 1
                    word_contexts[word].add((left_context, right_context))

                word_count += len(sentence)
                if word_count >= chunk_size:
                    update_finished = False
                    break

            print("... Building trie")
            trie = CTrie()
            for word, count in tqdm(word_counter.items()):
                try:
                    trie.add_word(word, count)
                except:
                    pass

            print("... Generating typo candidates")
            morph = pymorphy2.MorphAnalyzer()
            typo_candidates = defaultdict(list)
            for word, count in tqdm(word_counter.most_common()):
                if len(word) < 4:
                    continue
                if count < 1000:
                    continue
                if len(re.findall(r'\d', word)) > 0:
                    continue
                if len(word) <= 5:
                    errors = 1
                elif len(word) <= 11:
                    errors = 2
                else:
                    errors = 3

                try:
                    results = set(trie.fuzzy_search(word, errors))
                    results = sorted(results, reverse=True, key=lambda result: result.count)

                    for result in results:
                        if (result.count * 10 < count and word != result.word and
                                    len(set(morph.normal_forms(result.word)) & set(morph.normal_forms(word))) == 0):
                            typo_candidates[result.word].append(word)
                except:
                    pass

            print("... Generating typos")
            for word, candidates in tqdm(typo_candidates.items()):
                for context in word_contexts[word]:
                    if context_count[context] > 10:
                        best_candidate = max(candidates, key=lambda candidate: context_words[context][candidate])
                        if context_words[context][best_candidate] > context_words[context][word]:
                            typos.append((best_candidate, word, context_words[context][word]))

            print("... Updating trie")
            for word, count in tqdm(word_counter.items()):
                try:
                    if count >= 4:
                        self.trie.add_word(word, count)
                except:
                    pass

            word_counter.clear()
            context_count.clear()
            context_words.clear()
            word_contexts.clear()

        print("... Building typos model")
        self.typos_model.fit(typos, 10)

        print("Spellchecker building complete")

    def generate_word_candidates(self, word):
        word_candidates = {word}

        if len(word) > 1 and (is_word(word) or is_word(switch_layout(word))):
            results = self.trie.fuzzy_search(word, 2)
            results = sorted(results, key=lambda result: (result.errors, -result.count))
            for result in results[:100]:
                result_word = result.word
                word_candidates.add(result_word)

        return word_candidates

    def beam_search(self, dictionary: Dictionary, query: Query):
        params = self.vertical_parameters.get(query.site.vertical, self.default_parameters)
        sentence = query.tokens
        language_model = dictionary.language_model
        language_model_vertical = dictionary.language_model_vertical[query.site.vertical]
        vertical_words = dictionary.valid_words_for_vertical[query.site.vertical]

        sentence_candidates = [([], 0.0)]
        for index, word in enumerate(sentence):
            if word in stopwords.words():
                sentence_candidates = [(sentence_candidate + [word], sentence_log_prob)
                                       for (sentence_candidate, sentence_log_prob) in sentence_candidates]
                continue

            word_candidates = self.generate_word_candidates(word)
            word_candidates = [(word_candidate, self.typos_model.get_typo_log_prob(word, word_candidate))
                               for word_candidate in word_candidates]
            word_candidates = [(word_candidate, word_log_prob * params.gamma)
                               for (word_candidate, word_log_prob) in word_candidates]

            new_sentence_candidates = []
            for sentence_candidate, sentence_log_prob in sentence_candidates:
                lm_log_prob = (params.theta * np.log(language_model_vertical.get_word_prob(word, sentence_candidate[-2:])) +
                               (1 - params.theta) * np.log(language_model.get_word_prob(word, sentence_candidate[-2:])))
                if word in vertical_words or lm_log_prob > params.threshold:
                    new_sentence_candidates.append((sentence_candidate + [word], sentence_log_prob))
                    continue

                for word_candidate, word_log_prob in word_candidates:
                    lm_log_prob = (
                    params.theta * np.log(language_model_vertical.get_word_prob(word_candidate, sentence_candidate[-2:])) +
                    (1 - params.theta) * np.log(language_model.get_word_prob(word_candidate, sentence_candidate[-2:])))
                    word_log_prob += lm_log_prob

                    new_sentence_candidates.append(
                        (sentence_candidate + [word_candidate], sentence_log_prob + word_log_prob))
            sentence_candidates = sorted(new_sentence_candidates, key=lambda pair: -pair[1])[:params.beam_size]
        return sentence_candidates

    def fix_typos(self, query: Query, dictionary: Dictionary) -> Query:
        try:
            sentence_candidates = self.beam_search(dictionary, query)
            query.tokens = sentence_candidates[0][0]
        except UnicodeEncodeError:
            pass

        return query


class BaseSpellchecker(Serializable, LoggerMixin):
    def __init__(self, loader: Loader):
        super().__init__(loader)
        self.loader = loader
        self.trie = CTrie()

    def fix_typos(self, query: Query, dictionary: Dictionary) -> Query:
        sentence = query.tokens
        for i in range(len(sentence)):
            if sentence[i] not in dictionary.valid_words_for_vertical[query.site.vertical]:
                try:
                    results = self.trie.fuzzy_search(sentence[i], 2)
                    if len(results) > 0:
                        sentence[i] = max(results, key=lambda result: result.count).word
                except:
                    pass
        query.tokens = sentence
        return query


class EnchantSpellchecker(Serializable, LoggerMixin):
    def fix_typos(self, query: Query, dictionary: Dictionary) -> Query:
        enchant_ru_dict = enchant.Dict("ru_RU")
        enchant_en_dict = enchant.Dict("en_EU")

        sentence = query.tokens
        for i in range(len(sentence)):
            if sentence[i] not in dictionary.valid_words_for_vertical[query.site.vertical]:
                if len(sentence[i]) > 0:
                    candidates = enchant_ru_dict.suggest(sentence[i]) + enchant_en_dict.suggest(sentence[i])
                    candidates = [candidate for candidate in candidates if ' ' not in candidate]
                    if len(candidates) > 0:
                        sentence[i] = candidates[0]
        query.tokens = sentence
        return query