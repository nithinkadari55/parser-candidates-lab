import sys, os, onnx
from onnx import helper, shape_inference

src, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
m = onnx.load(src)
try:
    mi = shape_inference.infer_shapes(m)
except Exception as e:
    print("shape inference failed", e); mi = m
g = m.graph
types = {}
for vi in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output):
    types[vi.name] = vi.type.tensor_type.elem_type
inits = {i.name for i in g.initializer}
dyn = {"input_ids"}
enc = {"encoder_hidden_states"}
S, Dn = [], []
for n in g.node:
    ins = [x for x in n.input if x]
    if any(x in dyn for x in ins):
        Dn.append(n); dyn.update(n.output)
    elif any(x in enc for x in ins):
        S.append(n); enc.update(n.output)
    else:
        Dn.append(n)
s_out = set(o for n in S for o in n.output)
frontier = []
for n in Dn:
    for x in n.input:
        if x in s_out and x not in frontier:
            frontier.append(x)
for o in g.output:
    if o.name in s_out and o.name not in frontier:
        frontier.append(o.name)
missing = [x for x in frontier if x not in types]
print("static nodes", len(S), "dynamic", len(Dn), "frontier", len(frontier), "missing types", missing)
need_init = set(x for n in S for x in n.input) & inits
const_nodes_needed = set(x for n in S for x in n.input) - s_out - inits - {"encoder_hidden_states", ""}
print("non-init non-static inputs of S:", const_nodes_needed)
encin = [i for i in g.input if i.name == "encoder_hidden_states"][0]
A = helper.make_graph(S, "enc_kv", [encin], [helper.make_tensor_value_info(x, types[x], None) for x in frontier],
                      [i for i in g.initializer if i.name in need_init])
ma = helper.make_model(A, opset_imports=m.opset_import, producer_name="split")
ma.ir_version = m.ir_version
onnx.save(ma, os.path.join(outdir, "dec_static.onnx"))
used_b = set(x for n in Dn for x in n.input)
b_inputs = [i for i in g.input if i.name in used_b] + [helper.make_tensor_value_info(x, types[x], None) for x in frontier]
B = helper.make_graph(Dn, "dec_dyn", b_inputs, list(g.output), [i for i in g.initializer if i.name in used_b])
mb = helper.make_model(B, opset_imports=m.opset_import, producer_name="split")
mb.ir_version = m.ir_version
onnx.save(mb, os.path.join(outdir, "dec_dyn.onnx"))
print("B inputs", [i.name for i in b_inputs][:5], "...")
