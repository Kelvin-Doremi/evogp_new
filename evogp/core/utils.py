import inspect
from typing import Callable
import torch

from .node import (
    FUNCS,
    FUNCS_NAMES,
    FUNCS_DISPLAY,
    Func,
    MAX_STACK,
    MAX_FULL_DEPTH,
    NType,
    SYMPY_MAP,
)


def dict2prob(prob_dict):
    # Probability Dictionary to Distribution Function
    assert len(prob_dict) > 0, "Empty probability dictionary"

    prob = torch.zeros(len(FUNCS))

    for key, val in prob_dict.items():
        assert (
            key in FUNCS_NAMES
        ), f"Unknown function name: {key}, total functions are {FUNCS_NAMES}"
        idx = FUNCS_NAMES.index(key)
        prob[idx] = val

    # normalize
    prob = prob / prob.sum()

    return prob


def to_numpy(li):
    for idx, e in enumerate(li):
        if type(e) == torch.Tensor:
            li[idx] = e.cpu().numpy()
    return li


def check_tensor(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device="cuda", requires_grad=False)
    else:
        x = x.to("cuda").detach().requires_grad_(False)
    return x


def str_tree(value, node_type, subtree_size):
    res = ""
    for i in range(0, subtree_size[0]):
        if (
            (node_type[i] == NType.UFUNC)
            or (node_type[i] == NType.BFUNC)
            or (node_type[i] == NType.TFUNC)
        ):
            res = res + FUNCS_NAMES[int(value[i])]
        elif node_type[i] == NType.VAR:
            res = res + f"x[{int(value[i])}]"
        elif node_type[i] == NType.CONST:
            res = res + f"{value[i]:.2f}"
        res += " "

    return res


def randint(size, low, high, dtype=torch.int32, device="cuda", requires_grad=False):
    random = low + torch.rand(size, device=device, requires_grad=requires_grad) * (
        high - low
    )
    return random.to(dtype=dtype)


def inspect_function(func):
    assert isinstance(func, Callable), "formula should be Callable"
    sig = inspect.signature(func)
    parameters = sig.parameters
    assert len(parameters) > 0, "formula should have at least one parameter"
    for name, param in parameters.items():
        assert (
            param.default is inspect.Parameter.empty
        ), f"formula should not have default parameters, but got {name}={param.default}"

    return list(parameters.keys())
