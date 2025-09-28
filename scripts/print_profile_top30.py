#!/usr/bin/env python3
import sys
import os
import pstats
import io
import contextlib

prof = 'logs-alns/short_run.prof'
if not os.path.exists(prof):
    print('ERROR: profiler output not found at', prof, file=sys.stderr)
    sys.exit(2)

stats = pstats.Stats(prof)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    stats.strip_dirs().sort_stats('cumtime').print_stats(30)
txt = buf.getvalue()
print(txt)
os.makedirs('logs-alns', exist_ok=True)
with open('logs-alns/top30.txt', 'w', encoding='utf-8') as f:
    f.write(txt)
