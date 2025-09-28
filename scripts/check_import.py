import importlib, sys, pkgutil
try:
    m = importlib.import_module('ALNSCode')
    print('ALNSCode loaded from', getattr(m, '__file__', getattr(m,'__path__',None)))
except Exception as e:
    print('ALNSCode import failed:', e)

names = [m.name for m in pkgutil.iter_modules() if m.name.lower().startswith('alns') or m.name.lower().startswith('problem')]
print('candidates start with alns/problem:', names)
