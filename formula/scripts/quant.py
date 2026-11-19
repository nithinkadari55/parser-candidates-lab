import os, sys, shutil
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
src, dst = sys.argv[1], sys.argv[2]
for n in ["encoder_model.onnx", "decoder_model.onnx"]:
    pre = os.path.join(dst, n + ".pre.onnx")
    quant_pre_process(os.path.join(src, n), pre, skip_symbolic_shape=True)
    quantize_dynamic(pre, os.path.join(dst, n), weight_type=QuantType.QInt8, per_channel=False)
    os.remove(pre)
shutil.copy(os.path.join(src, "tokenizer.json"), dst)
for n in os.listdir(dst):
    print(n, os.path.getsize(os.path.join(dst, n)))
