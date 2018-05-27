from typing import List, Set, Dict, IO, Tuple, DefaultDict
from collections import Counter, defaultdict
from .serializable import Serializable
from .loader import Loader
from vypryamitel.fuzzy_set import FuzzySet
from vypryamitel.langutils import is_word, split_clear_query, check_and_touch_file
from common.site_config import SiteConfig
from warnings import warn
from language_model.language_model import LanguageModel, CStorage
import re
import string
from tqdm import tqdm

stoptrans = str.maketrans({p: ' ' for p in string.punctuation})


class Dictionary(Serializable):
    """
    Class that provides high-level dictionary interface.
    I.e. it can tell whether some word is "good", find nearest fuzzy neighbours, and contain fasttext logic.
    """

    def __init__(self, loader: Loader):
        super().__init__(loader)
        self.loader = loader
        self.fuzzy_sets = dict()
        # This dictionary contains words that appeared in site feeds (key is API key)
        self.valid_words_for_site = dict()  # type: Dict[str, Set[str]]
        # This dictionary contains words that appeared in site feeds, vertical-specific dictionary or vertical brand map
        self.valid_words_for_vertical = dict()  # type: Dict[str, Set[str]]
        self.language_model_vertical = dict()
        self.language_model = LanguageModel(CStorage())

    def update(self, verticals, update_russian_language_model=True):
        for vertical in verticals:
            vertical_feed_words = set()
            for site in self.loader.get_vertical_sites(vertical=vertical):
                site_feed_words = self.loader.get_site_dataset_by_config(site, feeds=True).get_all_words_from_feed()
                self.valid_words_for_site[site.name] = site_feed_words
                vertical_feed_words |= site_feed_words
            self.valid_words_for_vertical[vertical] = (vertical_feed_words |
                                                       self.loader.get_vertical_dictionary(vertical) |
                                                       self.loader.get_vertical_brands_set_words(vertical)) - \
                                                      self.loader.blacklist
            self.fuzzy_sets[vertical] = self._create_fuzzy_set(vertical=vertical)
            sentence_generator = self._sentence_generator([vertical], False)
            self.language_model_vertical[vertical] = self._create_language_model(sentence_generator)

        if update_russian_language_model:
            sentence_generator = self._sentence_generator([], True)
            self.language_model = self._create_language_model(sentence_generator)

    def has(self, item: str, site: SiteConfig) -> bool:
        """
        This function returns whether particular words appeared in this vertical sites' feeds, vertical whitelist, 
        vertical brand map, or general dictionary.
        """
        return (item in self.valid_words_for_vertical[site.vertical] or item in self.loader.general_dictionary)

    def _create_fuzzy_set(self, vertical: str):
        fuzzy = FuzzySet(use_levenshtein=False)
        for word in set(filter(is_word, (self.valid_words_for_vertical[vertical]))):
            fuzzy.add(word)
        return fuzzy

    def _sentence_generator(self, verticals: list, use_wiki: bool):
        delimiter = re.compile('[!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~]*')
        banned_substring_0 = re.compile('[\d\s]*')
        banned_substring_1 = re.compile('.*category=\d*\s*$')
        banned_substring_2 = re.compile('.*?utm_source=.*')
        banned_substring_3 = re.compile('.*http.*')

        generators = []
        for vertical in verticals:
            generators.append(self.loader.vertical_sentence_generator(vertical))
        if use_wiki:
            generators.append(self.loader.russian_sentence_generator())

        for generator in generators:
            for sentence in generator:
                if banned_substring_0.fullmatch(sentence):
                    continue
                if banned_substring_1.fullmatch(sentence):
                    continue
                if banned_substring_2.fullmatch(sentence):
                    continue
                if banned_substring_3.fullmatch(sentence):
                    continue
                if len(sentence) > 1000:
                    continue
                sentence = delimiter.sub('', sentence.strip().lower()).split(' ')
                sentence = [word for word in sentence if 0 < len(word) < 30]
                if len(sentence) == 0:
                    continue
                yield sentence

    def _create_language_model(self, sentence_generator):
        storage = CStorage()
        for sentence in tqdm(sentence_generator):
            try:
                storage.add_sentence(sentence, 3)
            except:
                pass

        storage.build_storage()
        return LanguageModel(storage)

    def _get_frequencies(self, sites: List[SiteConfig],
                         searches=False, feeds=False, file_searches_name=None, file_feeds_name=None) -> Dict[str, int]:
        def get_freq(sentences):
            words = set()
            for s in sentences:
                words.update(s)
            word_counter = Counter(words)
            for key in word_counter.keys():
                word_counter[key] = word_counter[key] - 1
            for s in sentences:
                for word in s:
                    word_counter[word] += 1
            return word_counter

        answer = dict()
        if searches:
            all_sentences = []
            sentences = []
            for site in sites:
                ds = self.loader.get_site_dataset_by_config(site=site, load_search_views=True)

                if ds.search_views is not None:
                    sentences += ds.search_views['searchstring'].tolist()
                all_sentences.extend([split_clear_query(s) for s in sentences])

            word_counter = get_freq(all_sentences)
            answer['searches'] = word_counter

            if file_searches_name is not None:
                check_and_touch_file(file_searches_name)
                with open(file_searches_name, 'w') as f:
                    f.write('word\tcount\n')
                    for word, count in word_counter.most_common():
                        if is_word(word):
                            f.write(word + '\t' + str(count) + '\n')
        if feeds:
            all_sentences = []
            sentences = []
            for site in sites:
                ds = self.loader.get_site_dataset_by_config(site=site, load_products=True,
                                                            load_categories=True)
                if ds.products is not None:
                    sentences = ds.products['name'].tolist()
                if ds.categories is not None:
                    sentences += ds.categories['name'].tolist()
                all_sentences.extend([split_clear_query(s) for s in sentences])

            word_counter = get_freq(all_sentences)
            answer['feeds'] = word_counter

            if file_feeds_name is not None:
                check_and_touch_file(file_feeds_name)
                with open(file_feeds_name, 'w') as f:
                    f.write('word\tcount\n')
                    for word, count in word_counter.most_common():
                        if is_word(word):
                            f.write(word + '\t' + str(count) + '\n')
        return answer

    def get_site_synonyms(self, site: SiteConfig) -> (DefaultDict[frozenset, str], Counter):
        df = self.loader._read_file_csv(self.loader.model_path + "manual-synonyms-" + site.vertical + ".csv")
        df = df.fillna('')
        synonyms_all = set(df[0].tolist()) | set(df[1].tolist())
        syn_dict = defaultdict(lambda: set())
        # (index, value from first col, value from second col, flag1, flag2)
        for _, a, b, flag1, flag2 in df.iloc[:, :4].itertuples():
            if flag1 == 2:
                if flag2 == '':
                    syn_dict[a].add(b)
                elif flag2 == '1':
                    syn_dict[b].add(a)
                elif flag2 == 3:
                    syn_dict[a].add(b)
                    syn_dict[b].add(a)

        syn_dict.pop("", None)
        ds = self.loader.get_site_dataset_by_config(site, load_products=True, load_search_views=True)

        if ds is None or ds.products is None or len(ds.products) == 0:
            self.loader.warning('Could not load products info for %s.' % site.name)
            return dict(), set()

        words = set([item for sublist in [s.split() for s in synonyms_all] for item in sublist])
        words = set(filter(is_word, words))
        words_from_synonyms_ids = defaultdict(lambda: set())

        for pid in ds.products.product_id.tolist():
            for word in ds.get_product_name(pid).lower().translate(stoptrans).split():
                if word in words:
                    words_from_synonyms_ids[word].add(pid)

        num_products_with_synonyms = Counter(
            {syn: len(set.intersection(*[words_from_synonyms_ids.get(s, set()) for s in syn.split(' ')])) for syn in
             synonyms_all})

        synonyms_all = set([s for s in synonyms_all if num_products_with_synonyms[s] > 0])

        if ds.search_views is not None and len(ds.search_views) > 100:
            words_from_synonyms_ids = defaultdict(lambda: set())
            for idx, search in ds.search_views.searchstring.iteritems():
                for word in str(search).lower().translate(stoptrans).split():
                    if word in words:
                        words_from_synonyms_ids[word].add(idx)

            popularity = Counter(
                {syn: len(set.intersection(*[words_from_synonyms_ids.get(s, set()) for s in syn.split(' ')])) for syn in
                 synonyms_all})

        else:
            self.loader.warning('Could not load searches info for %s.' % site.name)
            popularity = num_products_with_synonyms

        groups_of_same_synonyms = defaultdict(lambda: set())
        for syn in synonyms_all:
            s = frozenset(syn.split(' '))
            groups_of_same_synonyms[s].add(syn)

        synonyms_one_from_group = set()
        for _, value in groups_of_same_synonyms.items():
            if len(value) != 0:
                synonyms_one_from_group.add(value.pop())

        syn_dict = {word: (synonyms & synonyms_one_from_group) for word, synonyms in syn_dict.items()}
        syn_dict = {k: s for k, s in syn_dict.items() if len(s) != 0}
        syn_dict_sets = defaultdict(lambda: set())
        for key in syn_dict.keys():
            syn_dict_sets[frozenset(key.split()) - self.loader.nltk_stopwords] |= syn_dict[key]

        return syn_dict_sets, popularity

    def get_plur_map(self, vertical: str) -> dict():
        from pymorphy2 import MorphAnalyzer
        morph = MorphAnalyzer()
        map_plur = dict()
        for site in self.loader.get_vertical_sites(vertical):
            for key in self.get_site_synonyms(site)[0]:
                parts = [morph.parse(word)[0] for word in key]
                key_plur = frozenset(
                    [part.inflect({'plur'}).word if part.inflect({'plur'}) is not None else part.word for part in
                     parts])
                map_plur[key_plur] = key
        return map_plur
