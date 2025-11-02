from .dense import DenseHead
from .soft_moe import SoftMoEHead
from .sparse_moe import SparseMoEHead
# (we'll add Sparse/Hard imports later)

def build_head(kind: str, **kwargs): #kwargs=keyword arguments
    k = kind.lower()
    if k == "dense":
        return DenseHead(kwargs["in_dim"], kwargs["width"], kwargs["num_classes"])
    if k == "softmoe":
        return SoftMoEHead(kwargs["in_dim"], kwargs["num_classes"], kwargs["num_experts"], kwargs["hidden_mult"], kwargs["temperature"], kwargs["dropout_p"])
    if k == "sparsemoe":
        return SparseMoEHead(kwargs["in_dim"], kwargs["num_classes"], kwargs["num_experts"],
                        kwargs.get("hidden_mult", 0.0625), kwargs.get("k", 2), kwargs.get("temperature", 1.0),
                        kwargs.get("dropout_p", 0.1),
                        kwargs.get("gate_input_dropout", kwargs.get("dropout_p", 0.0)),
                        kwargs.get("gate_logits_dropout", kwargs.get("dropout_p", 0.0)),
                        kwargs.get("importance_coef", 0.1),
                        kwargs.get("load_coef", 0.1))
    if k == "hardmoe":
        raise NotImplementedError("Switch implementation of HardMoEHead coming later.")
    raise ValueError(f"Unknown head kind: {kind}")

