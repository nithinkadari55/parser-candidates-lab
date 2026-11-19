import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MFR_VARIANTS = ["p2t15_fp32", "p2t15_fp16", "p2t15_int16", "p2t15_int8", "p2t15_int4_b32", "p2t15_int4_b128",
                "p2t15_int4_b128_int8rest"]
REF = ["pp_formulanet_plus_s", "unirec_0_1b", "doctr_formulanet", "doctr_formulanet_int8"]
ALL = MFR_VARIANTS + ["p2t15_int8_split"] + REF


def load(name, threads=4):
    if name == "p2t15_int8_split":
        from mfr import MFRFast
        return MFRFast("p2t15_int8", threads)
    if name in MFR_VARIANTS:
        from mfr import MFR
        return MFR(name, threads)
    from refmodels import PPFormula, UniRec, DoctrFormula
    if name == "pp_formulanet_plus_s":
        return PPFormula(threads)
    if name == "unirec_0_1b":
        return UniRec(threads)
    if name == "doctr_formulanet":
        return DoctrFormula(threads, int8=False)
    if name == "doctr_formulanet_int8":
        return DoctrFormula(threads, int8=True)
    raise KeyError(name)
