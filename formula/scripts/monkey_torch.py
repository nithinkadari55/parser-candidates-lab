"""MonkeyOCRv2-S formula model (UniMERNet-T decoder) run through the official code on CPU (PyTorch)."""
import os, sys
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MK = os.environ.get("MONKEY_REPO", "/tmp/MonkeyOCRv2/formula")
WD = os.path.join(ROOT, "models_monkey")


@dataclass
class _Args:
    cfg_path: str
    options: list


def build(threads=4):
    import importlib
    import torch
    torch.set_num_threads(threads)
    sys.path.insert(0, MK)
    cwd = os.getcwd()
    os.chdir(MK)
    for m in ("unimernet.datasets.builders", "unimernet.models", "unimernet.processors", "unimernet.tasks"):
        importlib.import_module(m)
    import unimernet.tasks as tasks
    from unimernet.common.config import Config
    from unimernet.processors import load_processor
    opts = [f"model.finetuned={os.path.join(WD, 'model_only_fp32.pth')}",
            f"model.model_config.model_name={os.path.join(WD, 'unimernet_tiny')}",
            f"model.model_config.monkey_encoder_path={os.path.join(WD, 'monkeyocrv2_vit')}",
            f"model.tokenizer_config.path={os.path.join(WD, 'unimernet_tiny')}"]
    cfg = Config(_Args(os.path.join(MK, "configs/eval/monkeyocrv2_s.yaml"), opts))
    model = tasks.setup_task(cfg).build_model(cfg)
    model.to(torch.device("cpu")).eval()
    pcfg = cfg.config.datasets.formula_rec_eval.vis_processor.eval
    proc = load_processor(pcfg.name, pcfg)
    os.chdir(cwd)
    return model, proc


class MonkeyTorch:
    def __init__(self, threads=4):
        self.model, self.proc = build(threads)
        self.tok = self.model.tokenizer.tokenizer if hasattr(self.model, "tokenizer") else None

    def __call__(self, path):
        import torch
        from PIL import Image
        with Image.open(path) as im:
            x = self.proc(im.convert("RGB"))
        x = torch.stack([x], 0) if isinstance(x, torch.Tensor) else x
        with torch.no_grad():
            out = self.model.generate({"image": x}, temperature=0.0)
        pred = out["pred_str"][0]
        ids = out["pred_ids"][0].tolist()
        runaway = 2 not in ids
        n = ids.index(2) if not runaway else len(ids)
        return pred, n, runaway or n >= 1535
