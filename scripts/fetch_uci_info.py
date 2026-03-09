"""
查询 UCI 数据集信息：任务名称、输入维度、输出维度、数据点数量
使用 ucimlrepo.fetch_ucirepo 接口获取
"""
from ucimlrepo import fetch_ucirepo
import numpy as np

# 要查询的 UCI 数据集 ID
UCI_IDS = [1, 9, 165, 186, 291, 409, 477]


def preprocess_dataset(problem_id, problem):
    """与 uci_sr.py 保持一致的预处理，确保维度准确"""
    if problem_id == 1:
        mapping = {"M": 0, "F": 1, "I": 2}
        problem.data.features = problem.data.features.copy()
        problem.data.features.loc[:, "Sex"] = problem.data.features["Sex"].map(mapping)
    elif problem_id == 9:
        mask = problem.data.features["horsepower"].notna()
        problem.data.features = problem.data.features.loc[mask].copy()
        problem.data.targets = problem.data.targets.loc[mask].copy()
    return problem


def main():
    print("=" * 80)
    print("UCI 数据集信息查询")
    print("=" * 80)

    results = []
    for uid in UCI_IDS:
        try:
            problem = fetch_ucirepo(id=uid)
            problem = preprocess_dataset(uid, problem)

            X = problem.data.features.to_numpy(dtype=np.float32)
            y = problem.data.targets.to_numpy(dtype=np.float32)

            n_samples = X.shape[0]
            input_dim = X.shape[1]
            output_dim = y.shape[1]

            name = problem.metadata.get("name", "Unknown")
            abstract = problem.metadata.get("abstract", "")[:80] + "..."

            results.append(
                {
                    "id": uid,
                    "name": name,
                    "abstract": abstract,
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "n_samples": n_samples,
                }
            )

            print(f"\n【ID {uid}】{name}")
            print(f"  简介: {abstract}")
            print(f"  输入维度: {input_dim}")
            print(f"  输出维度: {output_dim}")
            print(f"  数据点数: {n_samples}")

        except Exception as e:
            print(f"\n【ID {uid}】获取失败: {e}")
            results.append(
                {
                    "id": uid,
                    "name": "Error",
                    "error": str(e),
                }
            )

    # 汇总表格
    print("\n" + "=" * 80)
    print("汇总表")
    print("=" * 80)
    print(f"{'ID':>4} | {'任务名称':<30} | {'输入维度':>8} | {'输出维度':>8} | {'数据点数':>8}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['id']:>4} | {'(获取失败)':<30} | {'-':>8} | {'-':>8} | {'-':>8}")
        else:
            print(
                f"{r['id']:>4} | {r['name']:<30} | {r['input_dim']:>8} | {r['output_dim']:>8} | {r['n_samples']:>8}"
            )


if __name__ == "__main__":
    main()
