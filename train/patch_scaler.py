import pathlib

p = pathlib.Path("/home/bloodycrown/train/omni_libs/OmniChar/core/src/inline_core/training/trainer.py")
s = p.read_text()

old = """            accelerator.backward(loss)
            # Bounded updates: the reference recipes (AI-Toolkit / musubi) all clip at 1.0."""
new = """            # Loss-scaled backward: the fp16 backward chain flushes gradient components
            # below ~6e-5 to zero before they ever reach the (fp32) LoRA grad buffers. Scaling
            # the loss by 65536 carries the small identity/caption-conditioned components
            # through the fp16 intermediates; unscale right after, then clip and step.
            _SC = 65536.0
            accelerator.backward(loss * _SC)
            for _p in transformer.parameters():
                if _p.grad is not None:
                    _p.grad.div_(_SC)
            if step < 30 or (step % 100) == 0:
                _lg = [(n, p.grad) for n, p in transformer.named_parameters()
                       if p.grad is not None and "lora_" in n]
                if _lg:
                    _e = sum(g.numel() for _, g in _lg)
                    _z = sum(int((g == 0).sum()) for _, g in _lg)
                    _sub = sum(int(((g != 0) & (g.abs() < 6.1e-5)).sum()) for _, g in _lg)
                    _mx = max(float(g.abs().max()) for _, g in _lg)
                    print(f"GRADLOG step {step} zero-frac {_z/_e:.5f} "
                          f"would-underflow-frac {_sub/_e:.5f} max {_mx:.6f}", flush=True)
            # Bounded updates: the reference recipes (AI-Toolkit / musubi) all clip at 1.0."""

assert old in s, "anchor not found"
p.write_text(s.replace(old, new))
print("trainer patched: scaler + GRADLOG")
