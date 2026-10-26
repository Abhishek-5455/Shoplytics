import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .rfm import build_rfm

OUTPUT_PATH = 'reports'

SEGMENT_NAMES = {
    0: "High-Spending Infrequent Buyers",
    1: "Repeat Buyers",
    2: "Recent One-Time Buyers",
    3: "Long-Inactive Buyers",
    4: "Moderately Inactive Buyers",
}

def scale_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    features = ['recency', 'frequency', 'monetary']

    if rfm.empty:
        raise ValueError("RFM data is empty")

    if rfm[features].isna().any().any():
        raise ValueError("RFM data contains null records")
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[features])

    return pd.DataFrame(
        scaled,
        columns=features,
        index=rfm.index
    )

def cluster_customers(rfm: pd.DataFrame, n_clusters = 5):
    if n_clusters < 2 or n_clusters >= len(rfm):
        raise ValueError("Number of clusters must be between 2 and (n_samples-1)")

    scaled = scale_rfm(rfm)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)

    result = rfm.copy()

    result['cluster'] = model.fit_predict(scaled)

    return result

def evaluate_clusters(
        rfm: pd.DataFrame, 
        cluster_count: tuple[int, ...] = (2,3,4,5,6,7)
    ) -> pd.DataFrame:
    SAMPLE_SIZE = 2000
    scaled = scale_rfm(rfm)

    sample = scaled.sample(n = SAMPLE_SIZE, random_state=42)

    result = []
    for cluster in cluster_count:
        model = KMeans(n_clusters=cluster, random_state=42, n_init=10)
        labels = model.fit_predict(sample)
        score = silhouette_score(sample, labels)
        result.append({
            "n_cluster": cluster,
            'score': score
        })

    return pd.DataFrame(result).sort_values('score', ascending=False).reset_index(drop=True)

def build_cluster_report(n_clusters: int = 5) -> pd.DataFrame:

    rfm = build_rfm()
    clustered = cluster_customers(rfm, n_clusters=n_clusters)

    report = (
        clustered.groupby("cluster")
        .agg(
            customer_count=("recency", "size"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            median_recency=("recency", "median"),
            median_frequency=("frequency", "median"),
            median_monetary=("monetary", "median"),
        )
        .round(2)
    )

    report["customer_percentage"] = (
        report["customer_count"] / report["customer_count"].sum() * 100
    ).round(2)

    return report


if __name__ == '__main__':
    rfm = build_rfm()
    score = evaluate_clusters(rfm)

    print('silhouette score:')
    print(score.round(4).to_string(index=False))

    best_k = int(score.loc[0]['n_cluster'])
    
    print(f"Best number of clusters : {best_k}")
    
    print('''
        Here, 2 cluster gives best silhouette score but choosing number of clusters as 5
        reveal distinct purchasing behaviour of the customers. It separates the frequent, infreqeunt buyers, 
        most recent and inactive buyers.
    ''')

    result = cluster_customers(rfm, n_clusters=5)
    result['segment_names'] = result['cluster'].map(SEGMENT_NAMES)

    print("cluster count")
    print(result['cluster'].value_counts().sort_index())

    print("\nCluster RFM profiles:")
    print(
        result.groupby("cluster")[
            ["recency", "frequency", "monetary"]
        ].mean().round(2)
    )

    report = build_cluster_report(n_clusters=5)

    print("Customer Segment Report:")
    print(report.to_string())

    result.to_csv(f"{OUTPUT_PATH}/customer_cluster_report.csv")
    report.to_csv(f"{OUTPUT_PATH}/customer_segment_report.csv")
    print(f"\nSaved both reports to {OUTPUT_PATH}")

    