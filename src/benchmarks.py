import argparse
import time
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class FaissANN:
    def __init__(self, num_clusters: int = 100, use_gpu: bool = False):
        self.num_clusters = num_clusters
        self.use_gpu = use_gpu
        self.index = None

    def train(self, embeddings):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        d = embeddings.shape[1]
        # faiss.IndexFlatIP
        # faiss.IndexIVFFlat
        # faiss.IndexHNSWFlat

        if self.num_clusters > 0:
            quantizer = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, self.num_clusters, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(d)

        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)

        self.index.train(embeddings)
        self.index.add(embeddings)

    def search(self, query_embeddings, k: int = 10):
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        faiss.normalize_L2(query_embeddings)
        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices


def validate_real(
    faiss_ann: FaissANN,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[float, float]:
    correct_predictions = 0
    total_predictions = len(test_df)

    begin = time.time()
    for emb, _, target in test_df.values:
        _, indices = faiss_ann.search([emb], k=1)
        predicted_target = train_df.iloc[indices[0][0]]["target"]
        if predicted_target == target:
            correct_predictions += 1

    total_time = time.time() - begin
    throughput = total_predictions / total_time

    return round(correct_predictions / total_predictions, 3), throughput


def run_validation_tests_for_real(
    path_to_dataframe: str | Path,
    num_clusters_list: List[int],
    test_size: float = 0.1,
    use_gpu: bool = False,
    embedding_column: str = "embedding",
):
    df = pd.read_parquet(path_to_dataframe)[[embedding_column, "intent", "target"]]

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    for num_clusters in num_clusters_list:
        faiss_ann = FaissANN(num_clusters=num_clusters, use_gpu=use_gpu)
        faiss_ann.train(train_df[embedding_column].tolist())
        accuracy, throughput = validate_real(faiss_ann, train_df, test_df)
        print(f"Number of clusters: {num_clusters}, accuracy: {accuracy}, throughput: {throughput}")


def validate_synthetic(
    faiss_ann: FaissANN,
    df: pd.DataFrame,
    msg: str = "",
) -> None:
    for emb, _, _ in tqdm(df.values, desc=msg):
        faiss_ann.search([emb], k=1)


def run_validation_tests_for_synthetic(
    path_to_dataframe: str | Path,
    num_clusters_list: List[int],
    size_limits_list: List[int],
    test_size: float = 0.1,
    use_gpu: bool = False,
    embedding_column: str = "embedding",
):
    df = pd.read_parquet(path_to_dataframe)[[embedding_column, "intent", "target"]]

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    for num_clusters, size_limit in zip(num_clusters_list, size_limits_list):
        faiss_ann = FaissANN(num_clusters=num_clusters, use_gpu=use_gpu)
        faiss_ann.train(train_df[embedding_column].tolist())
        device = "GPU" if use_gpu else "CPU"
        validate_synthetic(
            faiss_ann=faiss_ann, df=test_df[:size_limit], msg=f"{device}, with clusters = {num_clusters}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-type", type=str, default="real")
    parser.add_argument("--embedding-column", type=str, default="embedding")
    args = parser.parse_args()

    embedding_column = args.embedding_column
    data_type = args.data_type

    path_to_root = Path(__file__).parent.parent
    path_to_data = path_to_root / "data"

    print(f"Running validation tests for: {data_type} data `{embedding_column}`...")

    if data_type == "real":
        run_validation_tests_for_real(
            path_to_dataframe=Path(path_to_data / "xlm-roberta-embeddings.parquet"),
            num_clusters_list=[0, 8, 16, 32, 64, 128],
            embedding_column=embedding_column,
            test_size=0.2,
        )

    elif data_type == "synthetic":
        run_validation_tests_for_synthetic(
            path_to_dataframe=Path(path_to_data / "xlm-roberta-synthetic-embeddings.parquet"),
            num_clusters_list=[0, 8, 16, 32, 64, 128],
            size_limits_list=[1000, 2000, 5000, 10000, 20000, 25000],
            embedding_column=embedding_column,
            test_size=0.2,
        )

    else:
        raise ValueError(f"Unknown data type: {data_type}")

    print("Done.")


if __name__ == "__main__":
    main()
