import typing
import fasttext
import gensim
import pandas as pd
import postgresql
from typing import IO, List, Tuple, Dict, DefaultDict, Type, TypeVar, Optional
from common.utils import LoggerMixin
from vypryamitel.dataset import SDataset, load_dataset_by_config
from vypryamitel.elastic import Elastic
from .utils import CachedProperty, CachedFunctionProperty
from .serializable import Serializable
from collections import defaultdict, Counter
from common.site_config import SiteConfig
import vypryamitel.config as config
from nltk.corpus import stopwords
from common.site_config import SiteConfigManager


class Loader(LoggerMixin):
    """
    This a class that contains low-level data management required for search.
    i.e. data loading/mangling, external (such as ElasticSearch) connection and config management,
    and other such stuff should go there.
    It also guarantees that we do not load large files twice.
    """

    def __init__(self, model_path: str, dataset_path: str = None,
                 ps_feed_url: str = None, elastic_host: str = None,
                 spark_master: str = None, HDFS_path: str = None):
        """
        :param   model_path: path to directory that contains manual files or cached serialized models
        :param dataset_path: path to directory with event datasets
        """
        if dataset_path is None:
            self.warning("Dataset path not supplied. Assuming model_path and dataset_path are the same.")
            dataset_path = model_path

        if ps_feed_url is None:
            self.warning("PostgresSQL URL not supplied. Trying to load from environment.")
        self.ps_feed_url = ps_feed_url or config.platform_postgres_url
        if self.ps_feed_url is None:
            self.warning("PosgresSQL URL cannot be found. DB-related functions are unavailable.")

        if elastic_host is None:
            self.warning("Elastic host is not supplied. Trying to load from environment.")
        self.elastic_host = elastic_host or config.elastic_host
        if self.elastic_host is not None:
            self.elastic = Elastic(self.elastic_host)
        else:
            self.warning("Cannot instantiate Elastic: host not supplied")
            self.elastic = None

        if HDFS_path is None:
            self.warning("HDFS path is not supplied. Trying to load from environment.")
        self.HDFS_path = HDFS_path or config.HDFS_path  # ds-server: '/home/uploader/HDFS'

        if spark_master is None:
            self.warning("Spark master is not supplied. Trying to load from environment.")
        self.spark_master = spark_master or config.spark_master  # ds-server: 'local[1]'
        if self.HDFS_path is None or self.spark_master is None:
            self.warning("Invalid spark config. Spark-related computations are unavailable.")
        self._field_map = {'searchTerm': 'searchstring'}
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.config_manager = SiteConfigManager(rr_yaml_path=config.rr_yaml_path,
                                                text_dataset_yaml_path=config.text_dataset_yaml_path,
                                                postgres_url=self.ps_feed_url, spark_url=self.spark_master)

    # Aux function for easier implementation of other (HDFS) data sources.
    def _read_file_lines(self, filename: str) -> List[str]:
        with open(filename) as f:
            for line in f:
                yield line.rstrip()

    def _read_file_csv(self, filename: str, header=None) -> pd.DataFrame:
        return pd.read_csv(filename, delimiter='\t', header=header)

    def _read_file_whole(self, filename: str) -> str:
        with open(filename) as f:
            s = f.read()
        return s

    def save(self, obj: Serializable, filename: str):
        """
        This function takes Serializable object (e.g. model), stores it in storage and assigns name <filename>. 
        """
        with open(self.model_path + filename, 'wb') as f:
            obj.save(f)

    T = TypeVar("T")

    def load(self, what_to_load: Type[T], filename: str) -> T:
        """
        This function takes Serializable class (e.g. model class), and retrieves object stored in <filename>. 
        """
        with open(self.model_path + filename, 'rb') as f:
            return what_to_load.load(self, f)

    @CachedProperty
    def sites(self) -> List[SiteConfig]:
        return [site for site in self.config_manager.sites.values() if site.platform_config is not None]

    @CachedProperty
    def _spark_session(self):
        try:
            from os import environ
            environ['PYSPARK_SUBMIT_ARGS'] = "--packages com.databricks:spark-avro_2.11:4.0.0 pyspark-shell"
            from pyspark import sql
            return sql.SparkSession.builder \
                .master(self.spark_master) \
                .getOrCreate()
        except ImportError:
            self.critical('Pyspark could not be imported.')
        except:
            self.exception('')
        return None

    @CachedFunctionProperty
    def get_products_view_count(self, site: SiteConfig, **kwargs) -> dict:
        """
            Function that returns products popularity by year, month, day, hour.

                :param  site:  SiteConfig
                :param  year:  str, year in YYYY format. Default: None.
                :param month:  str, month in MM format.  Default: None.
                :param   day:  str,   day in DD format.  Default: None.
                :param  hour:  str,  hour in HH format.  Default: None.

                :returns    :  dict {productId: str -> nviews: int}
        """
        if self._spark_session is None:
            return {}

        result = {}
        try:
            df = self._spark_session.read \
                .format('com.databricks.spark.avro') \
                .load(self.HDFS_path + 'topics/productviews/')
            df = df.filter(df.siteId == site.platform_config.id)
            for var in ('year', 'month', 'day', 'hour',):
                val = kwargs.get(var)
                df = df.filter(df[var] == val) if val else df
            result = df.groupby('productId').count() \
                .toPandas().set_index('productId')['count'] \
                .to_dict()
        except:
            self.exception('')
        finally:
            return result

    @CachedFunctionProperty
    def get_hdfs_searches_by_config(self, site: SiteConfig, include_timestamps: bool = False, \
                                    limits: Optional[Tuple[Optional[int], Optional[int]]] = None) \
            -> pd.core.frame.DataFrame:
        """
        Function that fetches searchterms and (optionally) corresponding timestamps from HDFS database.

            :param site: SearchSiteConfig
            :param limits: tuple of minimal and maximum timestamp,
                                if timestamps should be returned as well.
                                One or both of the limits can be None. Defaults to None.

            :returns : pd.DataFrame with columns 'searchstring', ['timestamp']

            Note:
                timestamp limits should be expressed in seconds.
                To obtain valid timestamp from date you can run:

                    >> dateinfo = {'year':1970, 'month':1, 'day':1, 'hour':0,}
                    >> from datetime import datetime, timezone, timedelta
                    >> msk_tz = timezone(timedelta(hours=3))
                    >> datetime.datetime(**dateinfo, tzinfo=msk_tz).timestamp()

                That will return valid timestamp for passing in the limits tuple.
        """
        hdfs_columns = ('searchTerm',) + ('timestamp',) if include_timestamps else ()
        daps_columns = [self._field_map.get(c, c) for c in hdfs_columns]
        search_views = pd.DataFrame(columns=daps_columns)

        if self._spark_session is None:
            return search_views
        try:
            df = self._spark_session.read \
                .format('com.databricks.spark.avro') \
                .load(self.HDFS_path + 'topics/searches/')
            df = df.filter(df.siteId == site.platform_config.id)

            if limits is not None:
                l, r = limits
                df = df.filter(df['timestamp'] >= l * 1000) if l else df
                df = df.filter(df['timestamp'] < r * 1000) if r else df

            search_views = df[hdfs_columns].toPandas()
            search_views.columns = daps_columns
        except:
            self.exception('')
        finally:
            return search_views

    @CachedFunctionProperty
    def get_hdfs_searches_by_vertical(self, vertical: str, include_timestamps: bool = False, \
                                      limits: Optional[Tuple[Optional[int], Optional[int]]] = None) \
            -> pd.core.frame.DataFrame:
        """
            Wrapper around `get_hdfs_searches_by_config`
            that returns single dataframe for all sites in the vertical.

            For reference use docstring of `get_hdfs_searches_by_config`.
        """
        return pd.concat([self.get_hdfs_searches_by_config(site, include_timestamps, limits)
                          for site in self.get_vertical_sites(vertical)]).reset_index(drop=True)

    def get_site_by_name(self, name: str) -> SiteConfig:
        return self.config_manager.get_site_by_name(name)

    def get_site_by_id(self, id: int) -> SiteConfig:
        return self.config_manager.get_site_by_platform_id(id)

    def get_site_by_prefix(self, prefix: str) -> SiteConfig:
        return self.config_manager.get_site_by_prefix(prefix)

    def get_site_by_apikey(self, apikey: str) -> SiteConfig:
        return self.config_manager.get_site_by_apikey(apikey)

    @CachedProperty
    def verticals(self) -> List[str]:
        return list(set(site.vertical for site in self.sites))

    def get_vertical_sites(self, vertical: str) -> List[SiteConfig]:
        return [site for site in self.sites if site.vertical == vertical]

    @CachedFunctionProperty
    def get_site_dataset_by_config(self, site: SiteConfig, feeds=False, **kwargs) -> SDataset:
        """
            :param feeds: if we need to load all tables from feeds.
            :param **kwargs: all keyword arguments that are valid for `dataset.load_dataset_by_config`
        """
        if site is None:
            self.warning('Got `None` as SiteConfig.')
            return
        if not (feeds or any(kwargs.values())):
            self.warning('Please tell me what do you need using at least one kwarg. (sitename:%s)' % (site.name))
            return
        # May return datasets with partial data
        return load_dataset_by_config(site, datasets_path=self.dataset_path, feeds=feeds, **kwargs)

    def get_site_dataset_by_id(self, siteid, **kwargs) -> SDataset:
        """ For keyword arguments list look at get_site_dataset_by_config. """
        return self.get_site_dataset_by_config(self.get_site_by_id(siteid), **kwargs)

    def get_site_dataset_by_name(self, prefix, **kwargs) -> SDataset:
        """ For keyword arguments list look at get_site_dataset_by_config. """
        return self.get_site_dataset_by_config(self.get_site_by_prefix(prefix), **kwargs)

    def get_site_dataset_by_name(self, name, **kwargs) -> SDataset:
        """ For keyword arguments list look at get_site_dataset_by_config. """
        return self.get_site_dataset_by_config(self.get_site_by_name(name), **kwargs)

    def russian_sentence_generator(self):
        for line in self._read_file_lines(self.model_path + 'parsed-ruwiki'):
            yield line

    def vertical_sentence_generator(self, vertical, use_manual_searchstrings=True, use_feeds=True,
                                         use_dataset_searchstrings=True, use_hdfs=False):
        if use_manual_searchstrings:
            for line in self._read_file_lines(self.model_path + 'manual-searchstrings-' + vertical):
                yield line
        vertical_sites = self.get_vertical_sites(vertical=vertical)
        if use_manual_searchstrings or use_feeds:
            for site in vertical_sites:
                dataset = self.get_site_dataset_by_config(site, load_products=use_feeds,
                                                          load_search_views=use_dataset_searchstrings)
                if dataset.products is not None:
                    for name in dataset.products.name.values:
                        if type(name) == str:
                            yield name
                if dataset.search_views is not None:
                    for searchstring in dataset.search_views.searchstring.values:
                        if type(searchstring) == str:
                            yield searchstring
        if use_hdfs and self._spark_session is not None:
            vertical_ids = [site.id for site in vertical_sites if site.id]
            df = self._spark_session.read \
                .format('com.databricks.spark.avro') \
                .load(self.HDFS_path + 'topics/searches/')
            iterator = df.filter(df.siteId.isin(vertical_ids)) \
                .toLocalIterator()
            # iterator yields namedTuples of column values even if only one column was selected
            for searchstring in map(lambda r: getattr(r, 'searchTerm'), iterator):
                if type(searchstring) == str:
                    yield searchstring

    @CachedProperty
    def additional_layout_dictionary(self) -> set:
        tokens = self._read_file_whole(self.model_path + "manual-layout-classifier-data.txt").split()
        return set(tokens) - {''}

    @CachedProperty
    def blacklist(self) -> set():
        return set(self._read_file_lines(self.model_path + "manual-dictionary-blacklist.txt"))

    @CachedProperty
    def general_dictionary(self) -> set:
        dictionary = set(self._read_file_lines(self.model_path + "manual-dictionary.txt"))
        return (set(dictionary) | self.nltk_stopwords) - {''}

    @CachedFunctionProperty
    def get_vertical_brands_dictionary(self, vertical: str) -> dict:
        brands_df = self._read_file_csv(self.model_path + "manual-brands-dictionary-" + vertical + ".txt").dropna()
        # replace symbol appeared during export from googledoc
        brands_df = brands_df.applymap(lambda s: str(s).replace('\u200e', ''))
        brands = dict()
        for _, rus, eng in brands_df.iloc[:, :2].itertuples():
            brands[tuple(str(rus).split())] = str(eng).split()
        return brands

    @CachedFunctionProperty
    def get_vertical_brands_set_words(self, vertical: str) -> set:
        words = set()
        for rus, eng in self.get_vertical_brands_dictionary(vertical).items():
            words.update(list(rus) + list(eng))
        return words

    @CachedFunctionProperty
    def get_vertical_dictionary(self, vertical: str) -> set:
        words = set(self._read_file_whole(self.model_path + "manual-dictionary-" + vertical + ".txt").split())
        return words - {''}

    def get_vertical_dictionary_contains_word(self, word: str, vertical: str) -> bool:
        return word in self.get_vertical_dictionary(vertical)

    @CachedFunctionProperty
    def get_vertical_typo_train_dataset(self, vertical: str) -> pd.DataFrame:
        return self._read_file_csv(self.model_path + "manual-typo-classifier-" + vertical + ".csv", header=0)

    @CachedFunctionProperty
    def get_site_manual_pairs(self, site: SiteConfig) -> Dict[str, str]:
        return {}

    @CachedFunctionProperty
    def get_fasttext_model(self, vertical: str):
        return fasttext.load_model(self.model_path + "model-fasttext-" + vertical + '.bin')

    @CachedFunctionProperty
    def get_word2vec_model(self, vertical: str):
        return gensim.models.KeyedVectors.load_word2vec_format(self.model_path + "model-fasttext-" + vertical + '.vec')

    @CachedProperty
    def nltk_stopwords(self) -> set():
        return set(stopwords.words('russian'))
