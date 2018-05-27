import re
from .query import Query
from .dictionary import Dictionary


class Tagger:
    def __init__(self, url_re='.*(\://|www|http|ftp|\.ru|\.com|\.net|\.рф).*', email_re='.*@.*'):
        self.url_re = re.compile(url_re)
        self.email_re = re.compile(email_re)

        self.psn_re = re.compile('\d+(\.\d+)?')  # psn - point separated number
        self.csn_re = re.compile('\d+(\,\d+)?')  # csn - comma separated number
        self.number_re = re.compile('\d+([\,\.]\d+)?')

        self.russian_word_re = re.compile('[а-яА-ЯёЁ]+')
        self.english_word_re = re.compile('[a-zA-Z]+')
        self.word_re = re.compile('[а-яА-ЯёЁa-zA-Z]+')

    def is_psn(self, token):
        return self.psn_re.fullmatch(token) is not None

    def is_csn(self, token):
        return self.csn_re.fullmatch(token) is not None

    def is_number(self, token):
        return self.number_re.fullmatch(token) is not None

    def is_russian(self, token):
        return self.russian_word_re.fullmatch(token) is not None

    def is_english(self, token):
        return self.english_word_re.fullmatch(token) is not None

    def is_word(self, token):
        return self.word_re.fullmatch(token) is not None

    def is_url(self, token):
        return self.url_re.fullmatch(token) is not None

    def is_email(self, token):
        return self.email_re.fullmatch(token) is not None

    def tag_query(self, query: Query, dictionary: Dictionary):
        brand_words = set(dictionary.loader.get_vertical_brands_set_words(query.site.vertical))
        vertical_words = set(dictionary.valid_words_for_vertical[query.site.vertical])
        site_words = set(dictionary.valid_words_for_site[query.site.name])

        tags = []
        for i in range(len(query.tokens)):
            tag = set()
            if self.is_url(query.tokens[i]):
                tag.add('url')
            if self.is_email(query.tokens[i]):
                tag.add('email')
            if self.is_number(query.tokens[i]):
                tag.add('number')
            if self.is_russian(query.tokens[i]):
                tag.add('russian_word')
            if self.is_english(query.tokens[i]):
                tag.add('english_word')
            if self.is_word(query.tokens[i]):
                tag.add('word')
            if query.tokens[i] in brand_words:
                tag.add('brand_word')
            if query.tokens[i] in site_words:
                tag.add('site_word')
            if query.tokens[i] in vertical_words:
                tag.add('vertical_word')
            if query.tokens[i] in dictionary.loader.nltk_stopwords:
                tag.add('stopword')
            tags.append(tag)

        return tags