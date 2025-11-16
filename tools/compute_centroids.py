# Compute work centroids and 93% confidence radii
# given a reference_embeddings.pkl created by make_embeds.
# First draft by Claude Sonnet 4.5 2024-12-08, prompted by alanngnet


import pickle
import numpy as np
from collections import defaultdict


def compute_centroids(embeddings_file: str, output_file: str) -> None:
    """Compute and save centroids for all works"""
    # Load embeddings
    with open(embeddings_file, "rb") as f:
        embeddings = pickle.load(f)

    # Group performances by work
    work_to_perfs = defaultdict(list)
    for perf_id in embeddings:
        work_id = perf_id.split(".")[0]
        work_to_perfs[work_id].append(perf_id)

    # Calculate centroids
    centroids = {
        work_id: np.mean([embeddings[pid] for pid in perf_ids], axis=0)
        for work_id, perf_ids in work_to_perfs.items()
    }

    # Calculate confidence radius for each work
    radii = {}
    for work_id, perf_ids in work_to_perfs.items():
        perf_embeds = np.array([embeddings[pid] for pid in perf_ids])
        centroid = centroids[work_id]

        # Normalize embeddings and centroid before computing cosine distance
        perf_embeds_norm = perf_embeds / np.linalg.norm(
            perf_embeds, axis=1, keepdims=True
        )
        centroid_norm = centroid / np.linalg.norm(centroid)

        # Cosine distance: 1 - dot product (now that vectors are normalized)
        distances = 1 - np.dot(perf_embeds_norm, centroid_norm)

        # Radius = mean + 1.5*std to capture ~93% of performances
        radii[work_id] = float(np.mean(distances) + 1.5 * np.std(distances))

    # Save centroids, radii, and work-to-performance mapping
    with open(output_file, "wb") as f:
        pickle.dump(
            {
                "centroids": centroids,
                "radii": radii,
                "work_to_perfs": dict(work_to_perfs),
            },
            f,
        )
    print(
        f"Saved centroids and radii for {len(centroids)} works to {output_file}"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(
            "Usage: python compute_centroids.py reference_embeddings.pkl work_centroids.pkl"
        )
        sys.exit(1)

    compute_centroids(sys.argv[1], sys.argv[2])
