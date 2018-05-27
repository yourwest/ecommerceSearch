from common.site_config import SiteConfig


class Query:
    """
    This is mutable container class. It is supposed to hold query itself, additional supplied parameters, and
    anything we might want to cache for this query. It should not contain any "clever" logic.
    """

    def __init__(self, tokens, session_id, site, prefixes=None, suffixes=None):
        self.site = site
        self.session_id = session_id

        self.tokens = tokens

        self.prefixes = prefixes
        if self.prefixes is None:
            self.prefixes = ['' for token in self.tokens]
        assert len(self.tokens) == len(self.prefixes)

        self.suffixes = suffixes
        if self.suffixes is None:
            self.suffixes = ['' for token in self.tokens]
        assert len(self.tokens) == len(self.suffixes)

    def glue(self):
        result = []
        for i in range(len(self.tokens)):
            if self.tokens[i] == '' and self.prefixes[i] == '' and self.suffixes[i] == '':
                continue

            result.append(self.prefixes[i] + self.tokens[i] + self.suffixes[i])

        return ' '.join(result)


def make_query(searchstring: str, session_id: str, site: SiteConfig):
    return Query(searchstring.lower().split(), session_id, site)
