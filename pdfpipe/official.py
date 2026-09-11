"""Load unmodified upstream pure functions without importing training/CLI stacks."""
import ast
from collections import OrderedDict, defaultdict
import pathlib
import sys
import types

def tatr_functions():
    import numpy as np
    from fitz import Rect
    vendor=pathlib.Path(__file__).resolve().parents[1]/'vendor'
    root=vendor/'sources/table-transformer/src'
    if not root.exists():
        root=vendor/'table-transformer-upstream/src'
    sys.path.insert(0,str(root))
    import postprocess
    source=root/'inference.py'
    tree=ast.parse(source.read_text())
    selected=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef)],type_ignores=[])
    namespace={'np':np,'Rect':Rect,'postprocess':postprocess,
               'OrderedDict':OrderedDict,'defaultdict':defaultdict}
    exec(compile(selected,str(source),'exec'),namespace)
    return types.SimpleNamespace(**namespace)
