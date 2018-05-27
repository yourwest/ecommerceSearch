from .loader import Loader
from collections import defaultdict
from .query import Query
from .serializable import Serializable


class BrandReplacer(Serializable):
    """
    Class that replaces brands in query
    """

    def __init__(self, loader: Loader):
        super().__init__(loader=loader)
        self.loader = loader
        self.brands_dictionary = dict()
        self.site_feed_words = defaultdict(set)

    def update(self, verticals):
        for vertical in verticals:
            self.brands_dictionary[vertical] = self.loader.get_vertical_brands_dictionary(vertical)
            for site in self.loader.get_vertical_sites(vertical):
                self.site_feed_words[site.name] = self.loader.get_site_dataset_by_config(site, feeds=True).get_all_words_from_feed()

    def replace_brands(self, query: Query) -> Query:
        """ Note: function updates tokens and confidence simultaneously. """
        if not query.site.vertical in self.brands_dictionary:
            return query
        site_brand_dict = self.brands_dictionary[query.site.vertical]
        for k in range(len(query.tokens) + 1, 0, -1):
            for i in range(0, len(query.tokens) - k + 1):
                sub = tuple(query.tokens[i:(i + k)])
                if k == 1 and sub[0] in self.site_feed_words[query.site.name]:
                    continue
                if sub in site_brand_dict:
                    query.tokens[i:(i + k)] = site_brand_dict[sub]
        return query
