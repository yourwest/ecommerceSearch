import os
import tempfile

import typing

from typing import List
from .loader import Loader
from vypryamitel.config import platform_hdfs_namenodes_url
from .utils import CachedProperty, CachedFunctionProperty
from .serializable import Serializable
import fasttext
import gensim
from hdfs.client import Client
import pandas as pd


class HDFSLoader(Loader):
    def __init__(self, model_path:str, dataset_path: str = None, ps_feed_url: str = None):
        super().__init__(model_path, dataset_path, ps_feed_url)
        self.client = Client(platform_hdfs_namenodes_url)

    def _read_file_lines(self, filename: str) -> List[str]:
        with self.client.read(filename, encoding='utf-8') as f:
            for line in f:
                yield line.rstrip()

    def _read_file_whole(self, filename: str) -> str:
        with self.client.read(filename, encoding='utf-8') as f:
            s = f.read()
        return s

    def _read_file_csv(self, filename: str, header=None) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.client.download(filename, tmp_dir + "/temporal")
            return pd.read_csv(tmp_dir + "/temporal", delimiter='\t', header=header)

    def save(self, obj: Serializable, filename: str):
        self.critical("Starting serializing")
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.critical("Acquired temp dir")
            with open(tmp_dir + "/temporal", "wb") as writer:
                self.critical("Starting serializing")
                obj.save(writer)
                self.critical("Stopped serializing")
            if self.client.status(self.model_path + filename, strict=False) is not None:
                self.client.delete(self.model_path + filename)
            self.critical("Starting uploading")
            self.client.upload(self.model_path + filename, tmp_dir + "/temporal")
            self.critical("Done uploading")

    T = typing.TypeVar("T")

    def load(self, what_to_load: typing.Type[T], filename: str) -> T:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.client.download(self.model_path + filename, tmp_dir + "/temporal")
            with open(tmp_dir + "/temporal", "rb") as reader:
                return what_to_load.load(self, reader)

    @CachedFunctionProperty
    def get_fasttext_model(self, vertical: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = "model-fasttext-" + vertical + '.bin'
            self.client.download(self.model_path + filename, tmp_dir + "/" + filename)
            return fasttext.load_model(tmp_dir + "/" + filename)

    @CachedFunctionProperty
    def get_word2vec_model(self, vertical: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = "model-fasttext-" + vertical + '.vec'
            self.client.download(self.model_path + filename, tmp_dir + "/" + filename)
            return gensim.models.KeyedVectors.load_word2vec_format(tmp_dir + "/" + filename)
