import os
import numpy as np
from warnings import warn
from scipy.spatial import distance
from pymystem3 import Mystem
from vypryamitel.langutils import sentence_to_normal


def throw_warning(msg, stacklevel=2):
    warn(msg, stacklevel=stacklevel)


def check_and_touch_file(path: str):
    """
    Note:
        Checks provided path and creates file with desired permissions.
    :param path: full path to file

    :return: None
    """
    if not os.path.exists(path):
        open(path, 'w').close()
        os.chmod(path, 0o666)


def phrase_to_vec(phrase, num_features, model):
    """
    Note:
        Returns average of all words' vectors in a given phrase.

    :param phrase: str

    :param num_features: int

    :param model: w2v model

    :return: float array

    """

    featureVec = np.zeros((num_features), dtype="float32")
    nwords = 0
    for word in phrase:
        nwords += 1
        featureVec = np.add(featureVec, np.array(model[word]))

    if (nwords > 0):
        featureVec = np.divide(featureVec, nwords)
    return featureVec


def phrase_similarity(phrase_1, phrase_2, num_features, model):
    """
    Note:
        Calculates Cosine similarity between two phrase vectors.

    :param phrase_1: str

    :param phrase_2: str

    :param num_features: int

    :param model: w2v model

    :return: float

    """

    phrase_1_vec = phrase_to_vec(phrase_1.split(), model=model, num_features=num_features)
    phrase_2_vec = phrase_to_vec(phrase_2.split(), model=model, num_features=num_features)
    return 1 - distance.cosine(phrase_1_vec, phrase_2_vec)

def get_categories_name_normal_dict(categories):
    m = Mystem()
    categories['normal_name'] = categories['name'].apply(str).apply(lambda s: sentence_to_normal(s, m))
    return categories.set_index("category_id")["normal_name"].to_dict()

def get_words_from_dict(dictionary: dict):
    words = set()
    for key, value in dictionary.items():
        words.update(key.split()).update(value.split())
    return words
