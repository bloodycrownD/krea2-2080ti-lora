import sys

from safetensors.torch import load_file, save_file

snap, out = sys.argv[1], sys.argv[2]
sd = load_file(snap)
w = {("transformer." + k): v.contiguous() for k, v in sd.items() if not k.endswith(".alpha")}
save_file(w, out)
print("converted", out, len(w), "tensors")
