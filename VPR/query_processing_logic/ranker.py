from typing import List
from common.utils import LoggerMixin
from .loader import Loader
from .query import Query
from .serializable import Serializable


class DummyRanker(Serializable):
    def __call__(self, query: Query, candidates: List[str]):
        return self.rank_candidates(query, candidates)

    def update(self):
        pass

    def rank_candidates(self, query: Query, candidates: List[str]):
        return candidates[:min(len(candidates), 25)]


class SearchRanker(DummyRanker):
    pass