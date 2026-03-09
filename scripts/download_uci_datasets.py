"""
下载 UCI 数据集到 datasets 目录
格式：{id}_features.csv 和 {id}_targets.csv，与 999_features/999_targets 一致
"""
import os
from ucimlrepo import fetch_ucirepo

# 项目根目录（scripts 的上一级）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UCI_IDS = [1, 9, 165, 186, 291, 409, 477]
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")


def preprocess_dataset(problem_id, problem):
    """与 uci_sr.py 保持一致的预处理"""
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
    os.makedirs(DATASETS_DIR, exist_ok=True)

    for uid in UCI_IDS:
        try:
            problem = fetch_ucirepo(id=uid)
            problem = preprocess_dataset(uid, problem)

            features_path = os.path.join(DATASETS_DIR, f"{uid}_features.csv")
            targets_path = os.path.join(DATASETS_DIR, f"{uid}_targets.csv")

            problem.data.features.to_csv(features_path, index=False)
            problem.data.targets.to_csv(targets_path, index=False)

            n_samples = len(problem.data.features)
            n_feat = problem.data.features.shape[1]
            n_tgt = problem.data.targets.shape[1]
            print(f"ID {uid} ({problem.metadata['name']}): {n_samples} 样本, {n_feat} 特征, {n_tgt} 目标 -> {features_path}, {targets_path}")

        except Exception as e:
            print(f"ID {uid} 下载失败: {e}")

    print("\n完成")


if __name__ == "__main__":
    main()
