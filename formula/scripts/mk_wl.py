import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter
from texnorm import tokenize, DROP, DROP_WITH_ARG, UNWRAP, SYN, OPS, ARITY
BS = chr(92)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = os.path.join(ROOT, 'data/unimer/x/UniMER-Test/')
c = Counter()
for s in ['spe', 'cpe', 'sce']:
    for t in tokenize(open(d + s + '.txt', encoding='utf-8').read()):
        if t.startswith(BS) and len(t) > 2:
            c[t] += 1
wl = {t for t, n in c.items() if n >= 2}
v = json.load(open(os.path.join(ROOT, 'models/breezedeus__pix2text-mfr-1.5/tokenizer.json'), encoding='utf-8'))['model']['vocab']
wl |= {k for k in v if k.startswith(BS) and len(k) > 2}
wl |= {x for x in list(DROP) + list(DROP_WITH_ARG) + list(UNWRAP) + list(SYN) + list(OPS) + list(ARITY) if x.startswith(BS)}
wl -= {BS + x for x in ['label', 'ref', 'cite', 'eqref', 'color', 'textcolor', 'hbox', 'mbox']}
json.dump(sorted(wl), open(os.path.join(ROOT, 'manifests/cmd_whitelist.json'), 'w'))
print(len(wl))
