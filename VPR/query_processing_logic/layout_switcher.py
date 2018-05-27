from typing import List

from .dictionary import Dictionary
from .query import Query
from .serializable import Serializable
from collections import defaultdict
from .loader import Loader
from vypryamitel.langutils import switch_layout, change_sim_symb
from vypryamitel.keyboard_switcher import KeyboardSwitcher


class LayoutSwitcher(Serializable):
    """
    Class that can fix layout if necessary
    """

    def __init__(self, loader: Loader):
        super().__init__(loader)
        self.switcher = dict()

    def update(self, verticals: List[str], dictionary: Dictionary):
        for vertical in verticals:
            switcher = KeyboardSwitcher(vertical=vertical, dictionary=dictionary)
            X, y, bigrams, trigrams = switcher.matrix_constructing()
            model_switcher, scorers = switcher.model_creating(X, y)
            # Dropping that Dictionary reference; otherwise model might take ~10gigs of space
            switcher.dictionary = None
            # ToDo: this is horrible; KeyboardSwitcher should keep internal state instead
            self.switcher[vertical] = (switcher, model_switcher, bigrams, trigrams)

    def replace_similar_symbols(self, query: Query, dictionary: Dictionary):
        for i in range(len(query.tokens)):
            if not dictionary.has(query.tokens[i], query.site):
                query.tokens[i] = change_sim_symb(query.tokens[i])
        return query

    def fix_layout(self, query: Query, dictionary: Dictionary) -> Query:
        if query.site.vertical not in self.switcher:
            return query
        for i in range(len(query.tokens)):
            if not dictionary.has(query.tokens[i], query.site):
                if len(query.tokens[i]) > 1:
                    swl_word = switch_layout(query.tokens[i])
                    if dictionary.has(swl_word, query.site):
                        query.tokens[i] = swl_word
                    else:
                        (switcher, model_switcher, bigrams, trigrams) = self.switcher[query.site.vertical]
                        query.tokens[i] = switcher.keyboard_switcher(query.tokens[i], model_switcher, bigrams, trigrams,
                                                                     probability_for_switch=0.75)['response']
        return query
