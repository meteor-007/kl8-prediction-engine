"""共现图网络特征：PageRank、特征向量中心度与 Louvain 社区发现（契约 R21、§3.9）。

将正值中心化共现矩阵视为加权无向图网络：
- 节点为 80 个号码，边权为 A_ij > 0 的正向伴生强度。
- PageRank: 衡量球号在近期抱团网络中的核心枢纽位置。
- 特征向量中心度 (Eigenvector Centrality): 衡量球号与其它核心球号的关联深度。
- Louvain 社区发现: 识别号码抱团簇（聚类标签、簇规模、簇内强度）。
- 伴生码 Top-K: 为每个号码提取与其最亲密的伴生球号。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from kl8.constants import N_NUM

try:
    import networkx as nx  # type: ignore[import-untyped]
    from networkx.algorithms.community import louvain_communities as nx_louvain  # type: ignore[import-untyped]
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

#: networkx 分支允许捕获的异常（P8-M7：禁止裸 except；覆盖收敛失败、空图、数值退化）。
_NX_ERRORS: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    ZeroDivisionError,
    np.linalg.LinAlgError,
)
if _HAS_NX:  # pragma: no branch —— 未安装 networkx 时无需登记其异常类型
    _NX_ERRORS = _NX_ERRORS + (
        nx.NetworkXError,
        nx.NetworkXException,
        nx.NetworkXUnbounded,
        nx.PowerIterationFailedConvergence,
    )


@dataclass(frozen=True)
class GraphResult:
    """共现图网络拓扑特征。"""

    pagerank: np.ndarray
    eigen: np.ndarray
    community_id: np.ndarray
    community_size: np.ndarray
    community_strength: np.ndarray
    companion: Mapping[int, tuple[tuple[int, float], ...]]
    backend: str
    warnings: tuple[str, ...]


def _min_max_scale(arr: np.ndarray) -> np.ndarray:
    """将一维数组缩放到 [0, 1] 区间。"""
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if max_v - min_v < 1e-12:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def build_graph(A: np.ndarray, *, min_weight: float = 0.0) -> object | None:
    """从中心化共现矩阵的正值部分构建无向加权图。

    @param A (80, 80) 中心化共现矩阵
    @param min_weight 边权入图阈值（默认 0.0）
    @return networkx.Graph 或 None（无 networkx 时）
    """
    if not _HAS_NX:
        return None
    G = nx.Graph()
    G.add_nodes_from(range(N_NUM))
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(A[i, j])
            if w > min_weight:
                G.add_edge(i, j, weight=w)
    return G


def pagerank(
    A: np.ndarray,
    *,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    backend: str = "auto",
) -> np.ndarray:
    """计算图节点的 PageRank 分值，归一化到 [0, 1]。

    @param A (80, 80) 共现矩阵
    @param alpha 阻尼系数
    @param max_iter 最大迭代次数
    @param tol 收敛容差
    @param backend 'auto' | 'networkx' | 'numpy_power'
    @return (80,) float64
    """
    use_nx = _HAS_NX and backend in ("auto", "networkx")
    if use_nx:
        try:
            G = build_graph(A)
            if G is not None and G.number_of_edges() > 0:
                pr_dict = nx.pagerank(G, alpha=alpha, weight="weight", max_iter=max_iter, tol=tol)
                raw = np.array([pr_dict.get(i, 0.0) for i in range(N_NUM)], dtype=np.float64)
                return _min_max_scale(raw)
        except _NX_ERRORS:
            # 网络拓扑算法在退化输入下可能抛数值/收敛异常，降级到 NumPy 方案
            pass

    # NumPy 幂迭代降级方案
    pos_a = np.maximum(0.0, A)
    deg = np.sum(pos_a, axis=1)
    dangling = deg < 1e-12
    p = np.zeros_like(pos_a)
    for i in range(N_NUM):
        if not dangling[i]:
            p[i, :] = pos_a[i, :] / deg[i]
        else:
            p[i, :] = 1.0 / N_NUM

    v = np.full(N_NUM, 1.0 / N_NUM)
    for _ in range(max_iter):
        v_next = alpha * (v @ p) + (1.0 - alpha) / N_NUM
        if np.sum(np.abs(v_next - v)) < tol:
            break
        v = v_next
    return _min_max_scale(v)


def eigenvector_centrality(
    A: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> np.ndarray:
    """计算特征向量中心度，归一化到 [0, 1]。"""
    if _HAS_NX:
        try:
            G = build_graph(A)
            if G is not None and G.number_of_edges() > 0:
                ec_dict = nx.eigenvector_centrality_numpy(G, weight="weight")
                raw = np.array([abs(ec_dict.get(i, 0.0)) for i in range(N_NUM)], dtype=np.float64)
                return _min_max_scale(raw)
        except _NX_ERRORS:
            # 网络拓扑算法在退化输入下可能抛数值/收敛异常，降级到 NumPy 方案
            pass

    # NumPy 谱分析降级
    pos_a = np.maximum(0.0, A)
    if np.all(pos_a < 1e-12):
        return np.zeros(N_NUM, dtype=np.float64)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(pos_a)
        leading_vector = np.abs(eigenvectors[:, -1])
        return _min_max_scale(leading_vector)
    except (np.linalg.LinAlgError, ValueError):
        return np.zeros(N_NUM, dtype=np.float64)


def louvain_communities(
    A: np.ndarray,
    *,
    seed: int,
    resolution: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """执行 Louvain 社区划分算法（契约 R21）。

    @param A (80, 80) 中心化共现矩阵
    @param seed 随机种子，确保确定性
    @param resolution 分辨率参数
    @return (community_id: (80,) int32, community_size: (80,) int32)
    """
    comm_id = np.full(N_NUM, -1, dtype=np.int32)
    comm_size = np.ones(N_NUM, dtype=np.int32)

    if _HAS_NX:
        try:
            G = build_graph(A)
            if G is not None and G.number_of_edges() > 0:
                comms = nx_louvain(G, weight="weight", resolution=resolution, seed=seed)
                # comms 为 set[int] 列表
                for idx, c_set in enumerate(comms):
                    c_size = len(c_set)
                    for node in c_set:
                        comm_id[node] = idx
                        comm_size[node] = c_size
                # 孤立点
                for i in range(N_NUM):
                    if comm_id[i] == -1:
                        comm_id[i] = len(comms)
                        comm_size[i] = 1
                return (comm_id, comm_size)
        except _NX_ERRORS:
            # 网络拓扑算法在退化输入下可能抛数值/收敛异常，降级到 NumPy 方案
            pass

    # 降级方案：按最大连接分配
    pos_a = np.maximum(0.0, A)
    cluster = np.argmax(pos_a, axis=1).astype(np.int32)
    counts = np.bincount(cluster, minlength=N_NUM)
    sizes = counts[cluster].astype(np.int32)
    return (cluster, sizes)


def companion_topk(
    A: np.ndarray,
    *,
    k: int = 3,
) -> dict[int, tuple[tuple[int, float], ...]]:
    """提取每个号码的前 K 个强伴生球号与归一化亲和度。

    @param A (80, 80) 中心化共现矩阵
    @param k 提取前 K 个伴生码（默认 3）
    @return 键为 1..80，值为 ((伴生号码 1-based, 亲和度), ...)
    """
    res: dict[int, tuple[tuple[int, float], ...]] = {}
    for i in range(N_NUM):
        row = A[i, :].copy()
        row[i] = -999.0  # 排除自身
        # 降序排列索引
        sorted_indices = np.argsort(-row)
        top_pairs: list[tuple[int, float]] = []
        max_affinity = max(1e-9, float(row[sorted_indices[0]]))
        for idx in sorted_indices[:k]:
            val = float(row[idx])
            affinity = max(0.0, val / max_affinity) if max_affinity > 0 else 0.0
            top_pairs.append((idx + 1, round(affinity, 4)))
        res[i + 1] = tuple(top_pairs)
    return res


def build_graph_features(
    A: np.ndarray,
    *,
    seed: int,
    k: int = 3,
) -> GraphResult:
    """构建共现图拓扑全部特征。

    @param A (80, 80) 中心化共现矩阵
    @param seed 随机种子（用于确定性 Louvain 划分）
    @param k 伴生码个数
    @return GraphResult
    """
    warnings: list[str] = []
    backend = "networkx" if _HAS_NX else "numpy_power"
    if not _HAS_NX:
        warnings.append("networkx 未安装，降级使用 NumPy 谱分析与幂迭代计算图特征")

    pr = pagerank(A, backend=backend)
    ec = eigenvector_centrality(A)
    cid, csize = louvain_communities(A, seed=seed)
    companions = companion_topk(A, k=k)

    # 计算簇内平均正向共现强度
    comm_strength = np.zeros(N_NUM, dtype=np.float64)
    pos_a = np.maximum(0.0, A)
    for c in np.unique(cid):
        nodes = np.where(cid == c)[0]
        if len(nodes) > 1:
            sub = pos_a[np.ix_(nodes, nodes)]
            # 排除对角线的平均值
            n_sub = len(nodes)
            avg_strength = float(np.sum(sub) / (n_sub * (n_sub - 1)))
            comm_strength[nodes] = avg_strength

    return GraphResult(
        pagerank=pr,
        eigen=ec,
        community_id=cid,
        community_size=csize,
        community_strength=comm_strength,
        companion=companions,
        backend=backend,
        warnings=tuple(warnings),
    )
