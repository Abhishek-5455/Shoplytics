import pandas as pd

from .cluster_customers import cluster_customers
from .rfm import build_rfm


def compare_segments(cluster_count = (2,4,5)):
    """This compares the mean RFM characteristics and customer proportions for each tested cluster count."""
    rfm = build_rfm()

    report = []
    for k in cluster_count:
        clustered = cluster_customers(rfm, k)

        profile = (
            clustered.groupby('cluster')
            .agg(
                customer_count=('recency', 'size'),
                avg_recency=('recency', 'mean'),
                avg_frequency=('frequency', 'mean'),
                avg_monetary=('monetary', 'mean')
            ).reset_index()
        )

        profile['customer_percentage'] = (
            profile['customer_count'] / len(rfm) * 100
        ).round(2)

        profile['n_clusters'] = k

        report.append(profile)

    return pd.concat(report, ignore_index=True)

if __name__ == '__main__':
    comparison = compare_segments()

    print(comparison.round(2).to_string(index=False))

    output_path = 'data/processed'
    comparison.to_csv(f"{output_path}/segment_comparison.csv", index=False)

    print(f"Report saved to {output_path}/segment_comparison.csv")
