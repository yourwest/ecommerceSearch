from ctrie import CTrie


class NGramStorage:
    def __init__(self, n):
        self.n = n
        self.trie = CTrie()

    def add_sentence(self, sentence):
        for i in range(len(sentence)):
            try:
                self.trie.add_sequence(sentence[i: i + self.n])
            except:
                pass

    def get_continuations_count(self, n_gram):
        return self.trie.get_continuations_count(n_gram)

    def get_unique_continuations_count(self, n_gram):
        return self.trie.get_unique_continuations_count(n_gram)

    def get_continuations(self, n_gram, n=10):
        return self.trie.get_continuations(n_gram, n)

    def sort_continuations(self):
        self.trie.sort_continuations()

    def __getitem__(self, ngram):
        return self.trie.get_prefix_count(ngram)