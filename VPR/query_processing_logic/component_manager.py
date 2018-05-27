import typing

from typing import Type

from .serializable import Serializable
from .spellchecker import Spellchecker
from .dictionary import Dictionary
from .ranker import SearchRanker
from .explorer import Explorer
from .brand_replacer import BrandReplacer
from .space_fixer import SpaceFixer
from .synonym_finder import SynonymFinder
from .typo_fixer import TypoFixer
from .layout_switcher import LayoutSwitcher
from .loader import Loader


class AnyQueryComponentManager:
    __SERIALIZABLE_MAP = {
        BrandReplacer: "brand_replacer.pickle",
        Dictionary: "dictionary.pickle",
        SpaceFixer: "space_fixer.pickle",
        SynonymFinder: "synonym_finder.pickle",
        TypoFixer: "typo_fixer.pickle",
        LayoutSwitcher: "layout_switcher.pickle",
        SearchRanker: "search_ranker.pickle",
        Explorer: "explorer.pickle",
        Spellchecker: "spellchecker.pickle",
    }

    class Exception(Exception):
        pass

    def __init__(self, loader: Loader):
        """
        This class manages instantiation and loading/saving
        :param loader:
        """

        self.dictionary = None       # type: Dictionary
        self.brand_replacer = None   # type: BrandReplacer
        self.space_fixer = None      # type: SpaceFixer
        self.synonym_finder = None   # type: SynonymFinder
        self.layout_switcher = None  # type: LayoutSwitcher
        self.spellchecker = None     # type: Spellchecker
        self.explorer = None         # type: Explorer
        self.ranker = None           # type: SearchRanker
        self.loader = loader         # type: Loader

    # Even more horrible code below
    def get_dictionary(self) -> Dictionary:
        if self.dictionary is None:
            self.dictionary = Dictionary(self.loader)
        return self.dictionary

    def get_brand_replacer(self) -> BrandReplacer:
        if self.brand_replacer is None:
            self.brand_replacer = BrandReplacer(self.loader)
        return self.brand_replacer

    def get_space_fixer(self) -> SpaceFixer:
        if self.space_fixer is None:
            self.space_fixer = SpaceFixer(self.loader)
        return self.space_fixer

    def get_synonym_finder(self) -> SynonymFinder:
        if self.synonym_finder is None:
            self.synonym_finder = SynonymFinder(self.loader)
        return self.synonym_finder

    def get_layout_switcher(self) -> LayoutSwitcher:
        if self.layout_switcher is None:
            self.layout_switcher = LayoutSwitcher(self.loader)
        return self.layout_switcher

    def get_spellchecker(self) -> Spellchecker:
        if self.spellchecker is None:
            self.spellchecker = Spellchecker(self.loader)
        return self.spellchecker

    def get_explorer(self) -> Explorer:
        if self.explorer is None:
            self.explorer = Explorer(self.loader)
        return self.explorer

    def get_ranker(self) -> SearchRanker:
        if self.ranker is None:
            self.ranker = SearchRanker(self.loader)
        return self.ranker

    def store(self, obj: Serializable):
        """
        This function takes Serializable object and stores it in storage in appropriate place 
        """
        if obj.__class__ not in self.__SERIALIZABLE_MAP:
            raise self.Exception("Class " + obj.__class__.__name__ + " not found in class-filename map")
        return self.loader.save(obj, self.__SERIALIZABLE_MAP[obj.__class__])

    T = typing.TypeVar("T")

    def retrieve(self, what: Type[T]) -> T:
        if what not in self.__SERIALIZABLE_MAP:
            raise self.Exception("Class " + what.__name__ + " not found in class-filename map")
        return self.loader.load(what, self.__SERIALIZABLE_MAP[what])
