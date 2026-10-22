"""Activation steering for the scorers (ported from the turnstile project, Arditi et al. 2024 recipe).

  ablate  : directional ablation at EVERY transformer block output  h <- h - (h·d̂) d̂   (bypasses the refusal gate)
  add     : activation addition at ONE block output                  h <- h + α d̂        (α < 0 pushes away from refusal)
Direction files are unit vectors of the model's hidden size (torch .pt). The hooks act on every position.
`apply_steering(model, vec_path, mode, layer, alpha)` returns a handle list; `remove_steering(handles)` undoes it.
"""
import torch


def _blocks(model):
    for attr in ("model.layers", "model.language_model.layers", "transformer.h", "model.transformer.h", "language_model.model.layers"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    raise RuntimeError("cannot find the transformer blocks of this model")


class _Ablate:
    def __init__(self, d):
        self.d = d

    def __call__(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        d = self.d.to(h.device, h.dtype)
        h2 = h - (h * d).sum(-1, keepdim=True) * d
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2


class _Add:
    def __init__(self, d, alpha):
        self.d, self.alpha = d, float(alpha)

    def __call__(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        d = self.d.to(h.device, h.dtype)
        h2 = h + self.alpha * d
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2


def load_direction(path):
    v = torch.load(path, map_location="cpu")
    if isinstance(v, dict):
        v = v.get("direction", next(iter(v.values())))
    v = torch.as_tensor(v, dtype=torch.float32).flatten()
    return v / v.norm()


def apply_steering(model, vec_path, mode, layer=None, alpha=0.0):
    d = load_direction(vec_path)
    blocks = _blocks(model)
    hidden = model.get_input_embeddings().weight.shape[1]
    if d.numel() != hidden:
        raise ValueError(f"direction has {d.numel()} dims, model hidden size is {hidden}")
    handles = []
    if mode == "ablate":
        for b in blocks:
            handles.append(b.register_forward_hook(_Ablate(d)))
        emb = model.get_input_embeddings()
        handles.append(emb.register_forward_hook(lambda m, i, o: o - (o * d.to(o.device, o.dtype)).sum(-1, keepdim=True) * d.to(o.device, o.dtype)))
    elif mode == "add":
        if layer is None:
            raise ValueError("add needs --steer_layer")
        handles.append(blocks[int(layer)].register_forward_hook(_Add(d, alpha)))
    else:
        raise ValueError(f"unknown steering mode {mode!r}")
    return handles


def remove_steering(handles):
    for h in handles:
        h.remove()


def add_cli(ap):
    ap.add_argument("--steer_vec", default=None, help="unit direction (.pt); enables steering")
    ap.add_argument("--steer_mode", default="ablate", choices=["ablate", "add"])
    ap.add_argument("--steer_layer", type=int, default=None, help="block index for mode=add (the direction's fit locus)")
    ap.add_argument("--steer_alpha", type=float, default=0.0, help="coefficient for mode=add (negative = away from the direction)")


def from_args(model, args):
    if not getattr(args, "steer_vec", None):
        return []
    handles = apply_steering(model, args.steer_vec, args.steer_mode, args.steer_layer, args.steer_alpha)
    print(f"steering: {args.steer_mode} {args.steer_vec}" + (f" layer {args.steer_layer} alpha {args.steer_alpha}" if args.steer_mode == "add" else " (all blocks + embeddings)"), flush=True)
    return handles
