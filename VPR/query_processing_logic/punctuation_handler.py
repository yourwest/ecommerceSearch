from .query import Query
from .tagger import Tagger
import re


class PunctuationHandler:
    def __init__(self, bow_punct='\(\[\{\<', eow_punct='\,\.\?\!\;\:\)\]\}\>',
                 delimiters='\,\.\?\!\:\;\(\)\{\}\[\]\<\>'):
        self.bow_re = re.compile('[' + bow_punct + ']*$') # bow - begin of word
        self.eow_re = re.compile('^[' + eow_punct + ']*') # eow - end of word

        self.prefix_re = re.compile('^\W*')
        self.suffix_re = re.compile('\W*$')

        self.delimiter_re = re.compile('[' + delimiters + ']+')

        self.tagger = Tagger()

    # removes punctuation from the beginning and the end of each token
    def handle_outer_puncuation(self, query):
        tokens = []
        prefixes = []
        suffixes = []

        for i in range(len(query.tokens)):
            if re.fullmatch('\W*', query.tokens[i]):
                if i > 0:
                    suffixes[-1] += query.prefixes[i] + query.tokens[i] + query.suffixes[i]
            else:
                prefix = self.prefix_re.findall(query.tokens[i])[0]
                suffix = self.suffix_re.findall(query.tokens[i])[0]
                prefixes.append(query.prefixes[i] + prefix)
                suffixes.append(suffix + query.suffixes[i])
                tokens.append(query.tokens[i][len(prefix) : len(query.tokens[i]) - len(suffix)])

        query.tokens = tokens
        query.prefixes = prefixes
        query.suffixes = suffixes

        return query

    # split tokens if they have punctuation inside them.  abc,def(ghi:jkl) -> abc, def( ghi: jkl)
    def handle_inner_puncuation(self, query):
        i = 0
        while i < len(query.tokens):
            if query.tokens[i] == '':
                i += 1
                continue
            if self.tagger.is_number(query.tokens[i]) or self.tagger.is_url(query.tokens[i]) or self.tagger.is_email \
                    (query.tokens[i]):
                i += 1
                continue

            subtokens = self.delimiter_re.split(query.tokens[i])
            subprefixes = ['' for token in subtokens]
            subprefixes[0] = query.prefixes[i]
            subsuffixes = self.delimiter_re.findall(query.tokens[i]) + [query.suffixes[i]]

            query.tokens = query.tokens[:i] + subtokens + query.tokens[i + 1:]
            query.prefixes = query.prefixes[:i] + subprefixes + query.prefixes[i + 1:]
            query.suffixes = query.suffixes[:i] + subsuffixes + query.suffixes[i + 1:]

            i += 1

        return query

    # moves eow punctuation from the begin of word to the end of previous one.
    # moves bow punctuation from the end of word to the begin of next one.
    # examples: 1) abra ,cadabra -> abra, cadabra
    #           2) )a( b ...c, d )e :( -> a (b... c, d) e:
    #           3) 12 %,[ abacaba ] :) -> 12%, [abacaba]:)
    def format_puncuation(self, query):
        for i in range(len(query.tokens)):
            if i > 0:
                match = self.eow_re.findall(query.prefixes[i])[0]
                query.prefixes[i] = query.prefixes[i][len(match):]
                query.suffixes[i - 1] = query.suffixes[i - 1] + match

                match = self.bow_re.findall(query.suffixes[i - 1])[0]
                query.suffixes[i - 1] = query.suffixes[i - 1][:len(query.suffixes[i - 1]) - len(match)]
                query.prefixes[i] = match + query.prefixes[i]

                if self.tagger.is_number(query.tokens[i - 1]) and query.suffixes[i - 1] == '':
                    if query.prefixes[i].startswith('%'):
                        query.prefixes[i] = query.prefixes[i][1:]
                        query.suffixes[i - 1] += '%'

        match = self.eow_re.findall(query.prefixes[0])[0]
        query.prefixes[0] = query.prefixes[0][len(match):]

        match = self.bow_re.findall(query.suffixes[-1])[0]
        query.suffixes[-1] = query.suffixes[-1][:len(query.suffixes[-1]) - len(match)]

        return query

    def replace_decimal_comma(self, query):
        for i in range(len(query.tokens)):
            if self.tagger.is_csn(query.tokens[i]):
                query.tokens[i] = query.tokens[i].replace(',', '.')

        return query