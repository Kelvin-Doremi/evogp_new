"""
BFGS/L-BFGS 常数优化器，用于对 GP 树中的常数进行局部优化。
在符号回归中，常数通常通过随机扰动探索，本模块使用数值优化方法（L-BFGS）来精细调整常数。

支持两种后端：
- "cpu": 使用 scipy.optimize.minimize(L-BFGS-B)，小数据量时通常更快
- "gpu": 使用 torch.optim.LBFGS，大数据量时利用 GPU 加速
"""

import numpy as np
import torch
from typing import Optional, Tuple, Literal

from evogp.tree import Tree
from evogp.tree.utils import NType, Func


# PyTorch 函数映射 (用于可微前向计算)
def _apply_ufunc(func_id: int, x: torch.Tensor) -> torch.Tensor:
    """应用一元函数"""
    if func_id == Func.SIN:
        return torch.sin(x)
    elif func_id == Func.COS:
        return torch.cos(x)
    elif func_id == Func.TAN:
        return torch.tan(x)
    elif func_id == Func.SINH:
        return torch.sinh(x)
    elif func_id == Func.COSH:
        return torch.cosh(x)
    elif func_id == Func.TANH:
        return torch.tanh(x)
    elif func_id == Func.LOG:
        return torch.log(torch.clamp(torch.abs(x), min=1e-10))
    elif func_id == Func.LOOSE_LOG:
        return torch.log(torch.clamp(torch.abs(x), min=1e-10))
    elif func_id == Func.EXP:
        return torch.exp(torch.clamp(x, min=-20, max=20))
    elif func_id == Func.INV:
        return torch.where(torch.abs(x) < 1e-10, torch.sign(x) * 1e10, 1.0 / x)
    elif func_id == Func.LOOSE_INV:
        return 1.0 / torch.where(torch.abs(x) < 1e-10, torch.sign(x) * 1e-10, x)
    elif func_id == Func.NEG:
        return -x
    elif func_id == Func.ABS:
        return torch.abs(x)
    elif func_id == Func.SQRT:
        return torch.sqrt(torch.clamp(x, min=1e-10))
    elif func_id == Func.LOOSE_SQRT:
        return torch.sqrt(torch.abs(x))
    else:
        raise NotImplementedError(f"UFUNC {func_id} not supported for constant optimization")


def _apply_bfunc(func_id: int, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """应用二元函数 (left op right)"""
    if func_id == Func.ADD:
        return left + right
    elif func_id == Func.SUB:
        return left - right
    elif func_id == Func.MUL:
        return left * right
    elif func_id == Func.DIV:
        return torch.where(
            torch.abs(right) < 1e-10,
            left,  # 避免除零
            left / right,
        )
    elif func_id == Func.LOOSE_DIV:
        denom = torch.where(torch.abs(right) < 1e-10, torch.sign(right) * 1e-10, right)
        return left / denom
    elif func_id == Func.POW:
        return torch.pow(torch.clamp(torch.abs(left), min=1e-10), right)
    elif func_id == Func.LOOSE_POW:
        return torch.pow(torch.clamp(torch.abs(left), min=1e-10), right)
    elif func_id == Func.MAX:
        return torch.maximum(left, right)
    elif func_id == Func.MIN:
        return torch.minimum(left, right)
    elif func_id in (Func.LT, Func.GT, Func.LE, Func.GE):
        # 比较函数返回 1 或 -1
        if func_id == Func.LT:
            return torch.where(left < right, 1.0, -1.0)
        elif func_id == Func.GT:
            return torch.where(left > right, 1.0, -1.0)
        elif func_id == Func.LE:
            return torch.where(left <= right, 1.0, -1.0)
        else:  # GE
            return torch.where(left >= right, 1.0, -1.0)
    else:
        raise NotImplementedError(f"BFUNC {func_id} not supported for constant optimization")


def _apply_tfunc(func_id: int, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """应用三元函数 (if a>0 then b else c)"""
    if func_id == Func.IF:
        return torch.where(a > 0, b, c)
    else:
        raise NotImplementedError(f"TFUNC {func_id} not supported")


def _torch_forward(
    inputs: torch.Tensor,
    constants: torch.Tensor,
    node_value: np.ndarray,
    node_type: np.ndarray,
    subtree_size: np.ndarray,
    const_indices: np.ndarray,
    output_len: int = 1,
) -> torch.Tensor:
    """
    使用 PyTorch 可微方式计算树的前向传播。
    constants 对应 const_indices 中指定位置的常数节点值。
    """
    tree_size = int(subtree_size[0])
    stack: list = []

    const_ptr = 0
    for i in range(tree_size - 1, -1, -1):
        t = int(node_type[i] & 0x7F)  # TYPE_MASK
        v = node_value[i]

        if t == NType.VAR:
            var_idx = int(v)
            stack.append(inputs[:, var_idx])
        elif t == NType.CONST:
            stack.append(constants[const_indices[const_ptr]].expand(inputs.shape[0]))
            const_ptr += 1
        elif t == NType.UFUNC:
            x = stack.pop()
            stack.append(_apply_ufunc(int(v), x))
        elif t == NType.BFUNC:
            # 与 CUDA 一致：先 pop 为左子，后 pop 为右子（栈顶是后处理的左子树）
            left = stack.pop()
            right = stack.pop()
            stack.append(_apply_bfunc(int(v), left, right))
        elif t == NType.TFUNC:
            # 与 CUDA 一致：先 pop 为 condition，再 then，再 else
            a = stack.pop()
            b = stack.pop()
            c = stack.pop()
            stack.append(_apply_tfunc(int(v), a, b, c))
        else:
            raise ValueError(f"Unknown node type {t}")

    result = stack.pop()
    if output_len > 1:
        # 多输出情况需要累加，此处简化处理
        result = result.unsqueeze(-1)
    return result


def _extract_const_info(
    node_value: np.ndarray,
    node_type: np.ndarray,
    subtree_size: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    提取树中常数节点的索引和初始值。
    按后序（求值顺序，从 tree_size-1 到 0）收集，确保与 _torch_forward 一致。
    返回: (const_positions, const_values)
    - const_positions: 常数在 node_value 中的位置（求值顺序）
    - const_values: 常数的当前值
    """
    tree_size = int(subtree_size[0])
    const_positions = []
    const_values = []

    for i in range(tree_size - 1, -1, -1):
        t = int(node_type[i] & 0x7F)
        if t == NType.CONST:
            const_positions.append(i)
            const_values.append(float(node_value[i]))

    return np.array(const_positions), np.array(const_values, dtype=np.float64)


def _optimize_scipy(
    tree: Tree,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    node_value: np.ndarray,
    node_type: np.ndarray,
    subtree_size: np.ndarray,
    const_positions: np.ndarray,
    const_values: np.ndarray,
    max_iter: int,
    tol: float,
    const_bounds: Optional[Tuple[float, float]],
) -> Tuple[Tree, float]:
    """scipy L-BFGS-B 后端，小数据量时通常更快"""
    from scipy.optimize import minimize

    inputs_t = inputs.cpu().float()
    labels_t = labels.cpu().float().unsqueeze(1) if labels.dim() == 1 else labels.cpu().float()

    def loss_and_grad(c: np.ndarray) -> Tuple[float, np.ndarray]:
        c_t = torch.tensor(c, dtype=torch.float32, requires_grad=True)
        pred = _torch_forward(
            inputs_t, c_t, node_value, node_type, subtree_size,
            np.arange(len(c)), tree.output_len,
        )
        mse = torch.mean((pred - labels_t) ** 2)
        mse.backward()
        return float(mse.item()), c_t.grad.numpy().astype(np.float64)

    bounds = [const_bounds] * len(const_values) if const_bounds else None
    result = minimize(
        loss_and_grad, const_values, method="L-BFGS-B", jac=True,
        bounds=bounds, options={"maxiter": max_iter, "ftol": tol},
    )

    optimized_value = node_value.copy()
    for idx, pos in enumerate(const_positions):
        optimized_value[pos] = result.x[idx]

    optimized_tree = Tree(
        tree.input_len, tree.output_len,
        torch.tensor(optimized_value, dtype=torch.float32, device=tree.node_value.device),
        tree.node_type, tree.subtree_size,
    )
    return optimized_tree, float(result.fun)


def _optimize_es(
    tree: Tree,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    node_value: np.ndarray,
    node_type: np.ndarray,
    subtree_size: np.ndarray,
    const_positions: np.ndarray,
    const_values: np.ndarray,
    max_iter: int,
    const_bounds: Optional[Tuple[float, float]],
) -> Tuple[Tree, float]:
    """
    CMA-ES 进化策略：使用 Forest.SR_fitness 批量评估，利用 CUDA 加速。
    无梯度，每代一次批量评估，通常比 BFGS 更快。
    """
    from evogp.tree import Forest

    n_const = len(const_values)
    low, high = (const_bounds[0], const_bounds[1]) if const_bounds else (-1e6, 1e6)
    device = inputs.device

    # CMA-ES 参数：lambda=4+3*log(n)，mu=lambda/2
    lambda_ = min(int(4 + 3 * np.log(n_const + 1)), 32)
    lambda_ = max(lambda_, 4)
    mu = max(1, lambda_ // 2)

    mean = np.array(const_values, dtype=np.float64)
    sigma = 0.3
    # 对角协方差 (sep-CMA)
    C = np.ones(n_const, dtype=np.float64)

    best_mse = float("inf")
    best_c = mean.copy()

    node_type_t = tree.node_type
    subtree_size_t = tree.subtree_size

    labels_ = labels.float().unsqueeze(1) if labels.dim() == 1 else labels.float()

    for _ in range(max_iter):
        # 采样 lambda_ 个个体：y_i ~ mean + sigma * sqrt(C) * N(0,I)
        Z = np.random.randn(lambda_, n_const).astype(np.float64)
        Y = mean + sigma * np.sqrt(C) * Z
        Y = np.clip(Y, low, high)

        # 构建 Forest：同一结构，不同常数，批量评估
        batch_nv = np.tile(node_value, (lambda_, 1)).astype(np.float32)
        batch_nv[:, const_positions] = Y.astype(np.float32)

        batch_node_value = torch.as_tensor(batch_nv, dtype=torch.float32, device=device).contiguous()
        batch_node_type = node_type_t.unsqueeze(0).repeat(lambda_, 1).contiguous()
        batch_subtree_size = subtree_size_t.unsqueeze(0).repeat(lambda_, 1).contiguous()

        forest = Forest(
            tree.input_len, tree.output_len,
            batch_node_value, batch_node_type, batch_subtree_size,
        )
        # SR_fitness 返回 MSE（CUDA 实现），regressor 中 fitness = -SR_fitness 表示适应度
        raw = forest.SR_fitness(inputs.contiguous(), labels_.contiguous(), use_MSE=True)
        mse_arr = raw.cpu().numpy().astype(np.float64)

        # 选择 top mu
        idx = np.argsort(mse_arr)[:mu]
        mean = np.mean(Y[idx], axis=0)
        best_idx = int(np.argmin(mse_arr))
        if mse_arr[best_idx] < best_mse:
            best_mse = float(mse_arr[best_idx])
            best_c = Y[best_idx].copy()

        # 更新 C：rank-mu 对角
        Z_sel = Z[idx]
        C = np.mean(Z_sel ** 2, axis=0) + 1e-10
        # sigma 简单衰减
        sigma *= 0.98

    optimized_value = node_value.copy()
    for idx, pos in enumerate(const_positions):
        optimized_value[pos] = best_c[idx]

    optimized_tree = Tree(
        tree.input_len, tree.output_len,
        torch.tensor(optimized_value, dtype=torch.float32, device=device),
        tree.node_type, tree.subtree_size,
    )
    return optimized_tree, float(best_mse)


def _optimize_gpu(
    tree: Tree,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    node_value: np.ndarray,
    node_type: np.ndarray,
    subtree_size: np.ndarray,
    const_positions: np.ndarray,
    const_values: np.ndarray,
    max_iter: int,
    const_bounds: Optional[Tuple[float, float]],
) -> Tuple[Tree, float]:
    """torch.optim.LBFGS 后端，大数据量时利用 GPU"""
    device = inputs.device
    inputs_t = inputs.float()
    labels_t = labels.float().unsqueeze(1) if labels.dim() == 1 else labels.float()

    c_tensor = torch.tensor(const_values, dtype=torch.float32, device=device, requires_grad=True)
    lbfgs_inner = min(20, max_iter)
    num_steps = max(1, (max_iter + lbfgs_inner - 1) // lbfgs_inner)
    optimizer = torch.optim.LBFGS([c_tensor], max_iter=lbfgs_inner, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        pred = _torch_forward(
            inputs_t, c_tensor, node_value, node_type, subtree_size,
            np.arange(len(const_values)), tree.output_len,
        )
        mse = torch.mean((pred - labels_t) ** 2)
        mse.backward()
        return mse

    for _ in range(num_steps):
        optimizer.step(closure)
        if const_bounds is not None:
            with torch.no_grad():
                c_tensor.clamp_(const_bounds[0], const_bounds[1])

    optimized_value = tree.node_value.clone()
    const_positions_t = torch.tensor(const_positions, device=device, dtype=torch.long)
    optimized_value[const_positions_t] = c_tensor.detach()

    optimized_tree = Tree(tree.input_len, tree.output_len, optimized_value, tree.node_type, tree.subtree_size)
    final_mse = float(torch.mean((optimized_tree.forward(inputs) - labels) ** 2).item())
    return optimized_tree, final_mse


def optimize_tree_constants(
    tree: Tree,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 100,
    tol: float = 1e-6,
    const_bounds: Optional[Tuple[float, float]] = (-1e6, 1e6),
    backend: Literal["auto", "cpu", "gpu"] = "auto",
    method: Literal["bfgs", "es"] = "bfgs",
) -> Tuple[Tree, float]:
    """
    优化树中的常数，最小化 MSE 损失。

    Args:
        tree: 待优化的 GP 树
        inputs: 输入数据 (N, D)
        labels: 标签 (N,) 或 (N, 1)
        max_iter: 最大迭代次数
        tol: 收敛容差（仅 BFGS）
        const_bounds: 常数的上下界
        backend: "auto"=数据量<500用cpu否则gpu（仅 BFGS）
        method: "bfgs"=L-BFGS 梯度优化, "es"=CMA-ES 进化策略（批量评估，通常更快）

    Returns:
        (优化后的树, 优化后的 MSE 损失)
    """
    node_value = tree.node_value.cpu().numpy()
    node_type = tree.node_type.cpu().numpy()
    subtree_size = tree.subtree_size.cpu().numpy()

    const_positions, const_values = _extract_const_info(node_value, node_type, subtree_size)

    if len(const_values) == 0:
        return tree, float(torch.mean((tree.forward(inputs) - labels) ** 2).item())

    if method == "es":
        return _optimize_es(
            tree, inputs, labels, node_value, node_type, subtree_size,
            const_positions, const_values, max_iter, const_bounds,
        )

    n_data = inputs.shape[0]
    if backend == "auto":
        backend = "cpu" if n_data < 500 else "gpu"

    if backend == "cpu":
        return _optimize_scipy(
            tree, inputs, labels, node_value, node_type, subtree_size,
            const_positions, const_values, max_iter, tol, const_bounds,
        )
    else:
        return _optimize_gpu(
            tree, inputs, labels, node_value, node_type, subtree_size,
            const_positions, const_values, max_iter, const_bounds,
        )
