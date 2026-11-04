"""PyInstaller runtime hook (runs at the start of every process of the app, multiprocessing workers included).

pymzml (asari reads mzML with it) looks for its OBO vocabulary files next to the executable when sys.frozen is set,
the layout of cx_Freeze; failing that it downloads them, from an address that no longer answers. In this app the
files are where pymzml keeps them outside a frozen program: in its own package folder. pymzml.obo is given a view of
sys without `frozen`, so it looks there."""
import importlib.abc
import sys
import types


class _SysWithoutFrozen(types.ModuleType):
    def __getattr__(self, name):
        if name == "frozen":
            raise AttributeError(name)
        return getattr(sys, name)


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):              # get_code, is_package, resource readers: the frozen importer's own
        return getattr(self._inner, name)

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        module.sys = _SysWithoutFrozen("sys")


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "pymzml.obo":
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                spec.loader = _Loader(spec.loader)
                return spec
        return None


sys.meta_path.insert(0, _Finder())
