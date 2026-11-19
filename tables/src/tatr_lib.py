import sys, os, ast
from collections import defaultdict
import paths

TATR_SRC=paths.TATR_SRC
import importlib.util
spec=importlib.util.spec_from_file_location("postprocess",os.path.join(paths.THIRD,"postprocess.py"))
postprocess=importlib.util.module_from_spec(spec); spec.loader.exec_module(postprocess)
from fitz import Rect
import numpy as np
src=open(os.path.join(TATR_SRC,"inference.py"),encoding="utf-8").read()
tree=ast.parse(src)
want={"iob","align_headers","refine_table_structure","objects_to_structures","structure_to_cells","cells_to_html"}
ns={"postprocess":postprocess,"Rect":Rect,"defaultdict":defaultdict,"np":np}
import xml.etree.ElementTree as ET; ns["ET"]=ET
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name in want:
        exec(compile(ast.Module(body=[node],type_ignores=[]),"inference.py","exec"),ns)
for k,v in ns.items():
    if k in want: v.__globals__.update(ns)
objects_to_structures=ns["objects_to_structures"]; structure_to_cells=ns["structure_to_cells"]
THR={"table":0.5,"table column":0.5,"table row":0.5,"table column header":0.5,"table projected row header":0.5,"table spanning cell":0.5,"no object":10}
ID2={0:'table',1:'table column',2:'table row',3:'table column header',4:'table projected row header',5:'table spanning cell',6:'no object'}
def outputs_to_objects(logits, boxes, img_size, id2=ID2):
    e=np.exp(logits-logits.max(-1,keepdims=True)); p=e/e.sum(-1,keepdims=True)
    lab=p.argmax(-1); sc=p.max(-1)
    w,h=img_size; objs=[]
    for l,s,b in zip(lab,sc,boxes):
        name=id2[int(l)]
        if name=="no object": continue
        cx,cy,bw,bh=b
        objs.append({"label":name,"score":float(s),"bbox":[float((cx-bw/2)*w),float((cy-bh/2)*h),float((cx+bw/2)*w),float((cy+bh/2)*h)]})
    return objs
def run_post(objs, tokens):
    structs=objects_to_structures(objs, tokens, THR)
    if not structs: return [],0
    return structure_to_cells(structs[0], tokens)
