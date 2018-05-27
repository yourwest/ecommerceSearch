from .loader import Loader
from .dictionary import Dictionary
from vypryamitel.langutils import strip_left, samelang, dividebylang
from .query import Query
from .serializable import Serializable
import numpy as np


class SpaceFixer(Serializable):
    """
    Class that can glue words / split them
    """

    def __init__(self, loader: Loader, max_len_qval=8, min_len_qval=3):
        super().__init__(loader)
        self.max_len_qval, self.min_len_qval = max_len_qval, min_len_qval

    def dividebylang(self, query: Query, dictionary: Dictionary):
        """ This function is here only because "only `SpaceFixer` got to fix spaces". """
        #dictionary.update_query(query, tokens=dividebylang(query.glue()).split())
        return query

    def fix_spaces(self, query: Query, dictionary: Dictionary):
        """ Note: function updates tokens and confidence simultaneously. """
        language_model = dictionary.language_model_vertical[query.site.vertical]
        dict_check = lambda token: dictionary.has(token, query.site)
        i = 0
        while i < len(query.tokens):
            # ATTEMPTING TO SPLIT LONG WORD INTO SMALLER ONES
            if len(query.tokens[i]) > self.max_len_qval and not dict_check(query.tokens[i]):
                splitted_word = [query.tokens[i]]
                splitted_score = [language_model.get_word_prob(query.tokens[i], query.tokens[i - 2:i])]
                for j in range(1, len(query.tokens[i])):
                    if dict_check(query.tokens[i][:j]) and dict_check(query.tokens[i][j:]):
                        new_score = [language_model.get_word_prob(query.tokens[i][:j], query.tokens[i - 2:i]),
                                     language_model.get_word_prob(query.tokens[i][j:], query.tokens[i - 1:i] + [query.tokens[i][:j]])]
                        if sum(map(np.log, splitted_score)) < sum(map(np.log, new_score)):
                            splitted_word = [query.tokens[i][:j], query.tokens[i][j:]]
                            splitted_score = new_score
                if len(splitted_word) == 2:
                    query.tokens = query.tokens[:i] + splitted_word + query.tokens[i + 1:]
                    query.prefixes = query.prefixes[:i] + [query.prefixes[i], ''] + query.prefixes[i + 1:]
                    query.suffixes = query.suffixes[:i] + ['', query.suffixes[i]] + query.suffixes[i + 1:]
            elif (i + 1 < len(query.tokens) and samelang(query.tokens[i], query.tokens[i + 1]) and
                  query.suffixes[i] == '' and query.prefixes[i + 1] == ''):
                concat_word = [query.tokens[i], query.tokens[i + 1]]
                concat_score = [language_model.get_word_prob(query.tokens[i], query.tokens[i - 2:i]),
                                language_model.get_word_prob(query.tokens[i + 1], query.tokens[i - 1:i] + [query.tokens[i]])]
                if dict_check(query.tokens[i] + query.tokens[i + 1]):
                    new_score = [language_model.get_word_prob(query.tokens[i] + query.tokens[i + 1], query.tokens[i - 2:i])]
                    if sum(map(np.log, concat_score)) < sum(map(np.log, new_score)):
                        concat_word = [query.tokens[i] + query.tokens[i + 1]]
                        concat_score = [new_score]
                if dict_check(query.tokens[i] + '-' + query.tokens[i + 1]):
                    new_score = [language_model.get_word_prob(query.tokens[i] + '-' + query.tokens[i + 1], query.tokens[i - 2:i])]
                    if sum(map(np.log, concat_score)) < sum(map(np.log, new_score)):
                        concat_word = [query.tokens[i] + '-' + query.tokens[i + 1]]
                        concat_score = [new_score]
                if len(concat_word) == 1:
                    query.tokens = query.tokens[:i] + concat_word + query.tokens[i + 2:]
                    query.prefixes = query.prefixes[:i] + [query.prefixes[i]] + query.prefixes[i + 2:]
                    query.suffixes = query.suffixes[:i] + [query.suffixes[i + 1]] + query.suffixes[i + 2:]
            i += 1

        return query
