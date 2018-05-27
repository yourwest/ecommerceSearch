from typing import List
from common.utils import LoggerMixin
from .loader import Loader
from .query import Query
from .serializable import Serializable


class Explorer(LoggerMixin, Serializable):
    def update(self, verticals, debug=False):
        l = self.loader
        if l.elastic is None:
            self.error('Can not update: there is no connection to elastic.')
            return
        for cfg in l.sites:
            if not cfg.vertical in verticals:
                continue
            ds = l.get_site_dataset_by_config(cfg, load_products=True)
            if ds.products is None:
                self.warning('Could not load products for %s.'%cfg.name)
                continue
            if not len(ds.products):
                self.warning('There are no products for %s. Index was not created.'%cfg.name)
                continue
            if debug:
                self.warning('ATTENTION to reduced products! site: %s'%cfg.name)
                ds.products = ds.products.iloc[:1000]
            index_name = cfg.prefix
            l.elastic.create_index(index_name, 'product', ['name', 'product_id'])
            entries = list(ds.products[['product_id', 'name']].dropna().T.to_dict().values())
            for e in entries:
                e['_id'] = '{!s}_{!s}'.format(index_name, e['product_id'])
            l.elastic.store_data(index_name, 'product', entries)

    def __call__(self, query: Query, debug: bool = False) -> List[str]:
        if self.loader.elastic is None:
            self.error('Can not get search %s: there is no connection to elastic.'%self.__class__.__name__)
            return []
        return self.get_candidates(query, debug)

    def get_candidates(self, query: Query, debug: bool = False) -> List[str]:
        index_name = query.site.platform_config.prefix
        if not self.loader.elastic.exists(index_name):
            self.warning('Did not find index for the site `%s`.'%index_name)
            return []
        vals = [(r.value['product_id'], r.value['name']) for r in self.loader.elastic.search(index_name,
                                                        multimatch_query=query.glue(),
                                                        multimatch_fields='name', size=30)]
        if len(vals) > 0:
            candidates, names = zip(*vals)
        else:
            candidates = []
        if debug:
            return candidates, names
        return candidates
