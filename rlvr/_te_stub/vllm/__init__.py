"""Shadow stub for the system-site-packages `vllm` on ALCF Sophia.

The conda-provided vllm/_C.abi3.so was built against a different torch ABI and
fails to dlopen (undefined symbol c10::impl::cow::materialize_cow_storage),
which crashes `import trl.trainer.grpo_trainer`. TRL treats vllm as available
(dist-info exists) and imports many `vllm.*` submodules at import time. This
package sits first on PYTHONPATH and fabricates any `vllm.*` submodule/attribute
on demand as an inert placeholder; GRPO is run with use_vllm=False so nothing
here is ever called.
"""
import importlib.abc, importlib.machinery, sys, types

class _Placeholder:
    def __init__(self, *a, **k):
        raise RuntimeError("vllm is stubbed out on Sophia (broken system build); see rlvr/_te_stub/vllm")

class _StubModule(types.ModuleType):
    __path__ = []  # behave as a package
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Placeholder

class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm" or fullname.startswith("vllm."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None
    def create_module(self, spec):
        return _StubModule(spec.name)
    def exec_module(self, module):
        pass

sys.meta_path.insert(0, _Finder())
sys.modules[__name__].__class__ = _StubModule
__version__ = "0.11.0"
