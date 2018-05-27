from typing import IO
import dill


class Serializable:
    """ Mixin that allows saving and loading objects from internal library. """
    loader_attributes = []

    def __init__(self, loader: 'Loader'):
        self._set_refs(loader)
        # some necessary code

    def _set_refs(self, loader):
        """ Function that gets all necessary references from loader's fields. """
        self.loader = loader

    def _del_refs(self):
        """ Function that deletes all references to loader's fields. """
        for attr in ['loader'] + self.loader_attributes:
            if hasattr(self, attr):
                delattr(self, attr)

    def save(self, f: IO[bytes]):
        loader = self.loader
        self._del_refs()
        dill.dump(self, f)
        self._set_refs(loader)

    @staticmethod
    def load(loader: 'Loader', f: IO[bytes]) -> 'Serializable':
        shadow_self = dill.load(f)
        shadow_self._set_refs(loader)
        return shadow_self