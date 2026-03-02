"""
异步优化池 (Optimization Pool)

类似 PySR 的设计：GP 主线程继续进化，优化线程在后台不断抓取有潜力的个体进行 BFGS 精修，
修好后塞回种群。常数优化不再阻塞遗传变异。
"""

import queue
import threading
from typing import Optional, List, Tuple

import torch

from evogp.tree import Tree
from .constant_optimizer import optimize_tree_constants


def _copy_tree(tree: Tree) -> Tree:
    """深拷贝树，避免与主线程共享张量"""
    return Tree(
        tree.input_len,
        tree.output_len,
        tree.node_value.clone(),
        tree.node_type.clone(),
        tree.subtree_size.clone(),
    )


class OptimizationPool:
    """
    异步常数优化池。

    - 主线程：提交待优化个体，取回已完成的优化结果
    - 工作线程：不断从队列取个体，BFGS 优化后放入结果队列
    """

    def __init__(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 100,
        backend: str = "auto",
        max_queue_size: int = 100,
        method: str = "bfgs",
    ):
        """
        Args:
            inputs: 输入数据 (N, D)
            labels: 标签
            max_iter: 最大迭代次数
            backend: "auto"/"cpu"/"gpu"（仅 method="bfgs" 时有效）
            max_queue_size: 待优化队列最大长度，超出时丢弃最旧的
            method: "bfgs" 或 "es"，es 通常更快
        """
        self.inputs = inputs
        self.labels = labels
        self.max_iter = max_iter
        self.backend = backend
        self.method = method
        self.max_queue_size = max_queue_size

        self._input_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._output_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self):
        """启动后台优化线程"""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self):
        """停止后台线程"""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

    def submit(self, tree: Tree, fitness: float):
        """
        提交个体到优化队列（非阻塞）。
        若队列已满则跳过。
        """
        try:
            tree_copy = _copy_tree(tree)
            self._input_queue.put_nowait((tree_copy, fitness))
        except queue.Full:
            pass

    def submit_batch(self, trees: List[Tree], fitnesses: torch.Tensor):
        """批量提交 top-k 个体"""
        for tree, fit in zip(trees, fitnesses.tolist()):
            self.submit(tree, fit)

    def get_optimized(self) -> List[Tuple[Tree, float]]:
        """
        非阻塞获取所有已完成的优化结果。
        返回 [(optimized_tree, fitness), ...]
        """
        results = []
        while True:
            try:
                opt_tree, fitness = self._output_queue.get_nowait()
                results.append((opt_tree, fitness))
            except queue.Empty:
                break
        return results

    def pending_count(self) -> int:
        """待优化队列中的数量"""
        return self._input_queue.qsize()

    def _worker_loop(self):
        """工作线程：不断取个体、优化、放入结果"""
        while not self._stop_event.is_set():
            try:
                tree, fitness = self._input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                opt_tree, mse = optimize_tree_constants(
                    tree,
                    self.inputs,
                    self.labels,
                    max_iter=self.max_iter,
                    backend=self.backend,
                    method=self.method,
                )
                opt_fitness = -mse
                self._output_queue.put((opt_tree, opt_fitness))
            except (NotImplementedError, ValueError, RuntimeError):
                # 优化失败则原样放回
                self._output_queue.put((tree, fitness))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
