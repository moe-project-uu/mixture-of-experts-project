from .dense import DenseHead
from .soft_moe import SoftMoEHead
# (we'll add Soft/Sparse/Hard imports later)

def build_head(kind: str, **kwargs):
    k = kind.lower()
    if k == "dense":
        return DenseHead(kwargs["in_dim"], kwargs["width"], kwargs["num_classes"])
    # placeholders for later:
    if k == "softmoe":
        return SoftMoEHead(kw["in_dim"], kw["num_classes"], kw["num_experts"], kw["hidden_mult"], kw["temperature"], kw["dropout_p"])
    if k == "sparsemoe":
        raise NotImplementedError("SparseMoEHead coming later.")
    if k == "hardmoe":
        raise NotImplementedError("HardMoEHead coming later.")
    raise ValueError(f"Unknown head kind: {kind}")
