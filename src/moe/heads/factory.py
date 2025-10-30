from .dense import DenseHead
from .soft_moe import SoftMoEHead
# (we'll add Sparse/Hard imports later)

def build_head(kind: str, **kw): #kw=keyword arguments
    k = kind.lower()
    if k == "dense":
        return DenseHead(kw["in_dim"], kw["width"], kw["num_classes"])
    if k == "softmoe":
        return SoftMoEHead(kwargs["in_dim"], kwargs["num_classes"], kwargs["num_experts"], kwargs["hidden_mult"], kwargs["temperature"], kwargs["dropout_p"])
    if k == "sparsemoe":
        raise NotImplementedError("SparseMoEHead coming later.")
    if k == "hardmoe":
        raise NotImplementedError("HardMoEHead coming later.")
    raise ValueError(f"Unknown head kind: {kind}")

