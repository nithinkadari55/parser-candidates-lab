"""RF-DETR-Large DocLayNet (HF nie3e/rf-detr-large-doclay @fecf537, Apache-2.0; transformers 5.22.2 RfDetrForObjectDetection,
torch CPU) on the same DocLayNet test / OmniDocBench English pages that heron INT8 saw in order run 44477997398.
Post-processing is the model's own (image processor: 704x704 resize, ImageNet normalisation; sigmoid, top-300 over
queries x classes, no NMS), kept down to score 0.05 for mAP.
usage: r4_layout.py {dln,odb} SHARD NSHARDS"""
import os, sys, json, time
from r4_common import *

REPO, REV = "nie3e/rf-detr-large-doclay", "fecf537ebc5788fe7d0df3c7b55c50b0ff59307b"
LIMIT = int(os.environ.get("TB_LIMIT") or 0)


def norm(label): return label.replace(" - ", "-")


def main():
    ds, k, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    proc = AutoImageProcessor.from_pretrained(REPO, revision=REV)
    model = AutoModelForObjectDetection.from_pretrained(REPO, revision=REV).eval()
    id2 = {int(a): norm(b) for a, b in model.config.id2label.items()}
    print("rf-detr labels", id2, "torch threads", torch.get_num_threads(), flush=True)
    import r3_detect as RD
    gen = RD.dln_pages(k, n) if ds == "dln" else RD.odb_pages(k, n)
    os.makedirs(R4OUT, exist_ok=True)
    of = os.path.join(R4OUT, f"rfdetr-{ds}-{k}.jsonl"); cnt = 0
    with open(of, "w", encoding="utf-8") as f, torch.inference_mode():
        for _, pid, img, W, H in gen:
            t = time.perf_counter()
            inp = proc(images=[img], return_tensors="pt")
            out = model(**inp)
            res = proc.post_process_object_detection(out, threshold=0.05, target_sizes=[(img.size[1], img.size[0])], top_k=300)[0]
            sec = time.perf_counter() - t
            sx, sy = W / img.size[0], H / img.size[1]
            dets = [[id2[int(l)], round(float(s), 4), round(float(b[0]) * sx, 1), round(float(b[1]) * sy, 1), round(float(b[2]) * sx, 1), round(float(b[3]) * sy, 1)]
                    for s, l, b in zip(res["scores"], res["labels"], res["boxes"])]
            f.write(json.dumps(dict(ds=ds, id=pid, W=W, H=H, sec=sec, dets=dets)) + "\n"); cnt += 1
            if cnt % 100 == 0: print(ds, k, cnt, round(sec, 2), "s/page", flush=True)
            if LIMIT and cnt >= LIMIT: break
    print("done", ds, k, cnt, flush=True)


if __name__ == "__main__":
    main()
