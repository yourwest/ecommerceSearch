from typing import List
from .query import Query
from .serializable import Serializable
from .loader import Loader
from .dictionary import Dictionary
from collections import defaultdict


class SynonymFinder(Serializable):
    """
    Class that analyses query and generates string with synonyms.
    """

    def __init__(self, loader: Loader):
        super().__init__(loader)
        self.loader = loader
        self.synonyms_dictionary = defaultdict(dict)
        self.frequencies_synonyms = defaultdict(dict)
        self.stopwords = set()
        self.plur_map = defaultdict(dict)

    def update(self, verticals: List[str], dictionary: Dictionary):
        self.stopwords = self.loader.nltk_stopwords
        for vertical in verticals:
            sites = self.loader.get_vertical_sites(vertical)
            for site in sites:
                self.synonyms_dictionary[site.name], self.frequencies_synonyms[
                    site.name] = dictionary.get_site_synonyms(site)
            self.plur_map[vertical] = dictionary.get_plur_map(vertical)

    def find_synonyms(self, query: Query, num=10) -> List[str]:
        toks = frozenset(set(query.tokens) - self.stopwords)
        toks = self.plur_map[query.site.vertical].get(toks, toks)
        ans = list(self.synonyms_dictionary[query.site.name].get(toks, set()))
        ans.sort(key=lambda s: -self.frequencies_synonyms[query.site.name].get(s, 0))
        return ans[:num]
