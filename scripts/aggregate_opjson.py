"""
Aggregate OPJSON entries from logs-alns/alns_optimization.log and print top operators by total elapsed time.
"""
import re, ast, json, collections
from pathlib import Path

log_path = Path("logs-alns") / "alns_optimization.log"
if not log_path.exists():
    print(f"Log not found: {log_path}")
    raise SystemExit(1)

pat = re.compile(r'OPJSON\s*(\{.*\})')
agg = collections.defaultdict(lambda: {'count': 0, 'total': 0.0, 'samples': []})

with log_path.open(encoding='utf-8', errors='replace') as fh:
    for line in fh:
        m = pat.search(line)
        if not m:
            continue
        s = m.group(1).strip()
        j = None
        # try safe literal_eval first (handles Python dict repr)
        try:
            j = ast.literal_eval(s)
        except Exception:
            # fallback to JSON with single-quote replacement
            try:
                j = json.loads(s.replace("'", '"'))
            except Exception:
                # try to recover truncated/escaped JSON by finding first '{' to last '}'
                try:
                    start = s.index('{')
                    end = s.rindex('}') + 1
                    snippet = s[start:end]
                    j = json.loads(snippet.replace("'", '"'))
                except Exception:
                    continue
        if not isinstance(j, dict):
            continue
        op = j.get('op', '<unknown>')
        try:
            t = float(j.get('elapsed_sec', 0.0) or 0.0)
        except Exception:
            t = 0.0
        ent = agg[op]
        ent['count'] += 1
        ent['total'] += t
        if len(ent['samples']) < 5:
            ent['samples'].append(j)

# print header
print("Top operators by total elapsed time (from OPJSON):")
print("{:40s} {:>8s} {:>12s} {:>10s}".format("op", "calls", "total_sec", "avg_sec"))
for op, v in sorted(agg.items(), key=lambda x: -x[1]['total'])[:30]:
    avg = v['total'] / v['count'] if v['count'] else 0.0
    print("{:40s} {:8d} {:12.3f} {:10.3f}".format(op, v['count'], v['total'], avg))

# optionally print small samples for top 5 ops
print("\\nSamples (up to 5) for top 5 operators:")
for op, v in sorted(agg.items(), key=lambda x: -x[1]['total'])[:5]:
    print(f"--- {op} (calls={v['count']}, total={v['total']:.3f}) ---")
    for s in v['samples']:
        print(" ", s)
