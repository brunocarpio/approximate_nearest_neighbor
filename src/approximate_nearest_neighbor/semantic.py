import re
import string

import matplotlib.pyplot as plt
import numpy as np
import tensorflow_hub as hub
from matplotlib.collections import LineCollection

from approximate_nearest_neighbor.graph import NSMWGraph, Space, Vertex

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)


def embed_text(text) -> np.ndarray:
    return model(text).numpy().astype(np.float32)


def preprocess(text: str) -> str:
    """Normalize text before embedding.

    Lowercases the text, drops newlines, strips punctuation and digits, and
    collapses runs of whitespace so semantically equal sentences embed alike.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def project_2d(*embeddings: np.ndarray) -> np.ndarray:
    """Project high-dimensional embeddings onto their two principal components.

    The embeddings are stacked so that the projection is fitted on all of them at
    once, which keeps the query in the same coordinate frame as the sentences.
    """
    stacked = np.vstack(embeddings)
    centered = stacked - stacked.mean(axis=0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    return centered @ components[:2].T


def visualize(
    graph: NSMWGraph,
    sentences: list[str],
    sentence_embeddings: np.ndarray,
    query: str,
    query_embedding: np.ndarray,
    results: list[tuple],
) -> None:
    """Plot the sentence graph, the query and the top-k nearest neighbors in 2D."""
    projected = project_2d(sentence_embeddings, query_embedding[None])
    points = projected[: len(sentences)]
    query_point = projected[-1]

    _, ax = plt.subplots(figsize=(12, 9))

    segments = [
        [points[v.id], points[n.id]]
        for v in graph.vertices.values()
        for n in v.neighbors
        if v.id < n.id
    ]

    if segments:
        ax.add_collection(
            LineCollection(
                segments,
                colors="steelblue",
                linewidths=0.6,
                alpha=0.12,
                zorder=1,
            )
        )

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=45,
        c="gray",
        alpha=0.8,
        label=f"{len(sentences)} sentences",
        zorder=2,
    )

    # Only the top-k hits and the query are labelled: with a large corpus the
    # unlabelled points convey the overall layout without cluttering the plot.
    result_ids = [vid for _, vid, _ in results]
    ax.scatter(
        points[result_ids, 0],
        points[result_ids, 1],
        s=170,
        facecolors="none",
        edgecolors="crimson",
        linewidths=2,
        label=f"top {len(result_ids)} neighbors",
        zorder=4,
    )

    # Stagger labels so nearby points don't overprint each other.
    label_offsets = [(9, 8), (-9, 8), (9, -18), (-9, -18)]
    for rank, (distance, vid, _) in enumerate(results, start=1):
        dx, dy = label_offsets[(rank - 1) % len(label_offsets)]
        ax.annotate(
            f"#{rank} (cos {distance:.3f}) {sentences[vid]}",
            (points[vid, 0], points[vid, 1]),
            fontsize=8.5,
            color="crimson",
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right",
            zorder=5,
        )

    ax.scatter(
        query_point[0],
        query_point[1],
        c="orange",
        s=240,
        marker="*",
        edgecolors="black",
        linewidths=0.5,
        label="query",
        zorder=6,
    )
    ax.annotate(
        f'query:\n"{query}"',
        (query_point[0], query_point[1]),
        fontsize=9,
        fontweight="bold",
        color="darkorange",
        xytext=(-12, -6),
        textcoords="offset points",
        ha="right",
        zorder=7,
    )

    ax.set_title(
        f'Nearest sentences to "{query}"\n'
        f"({len(sentences)} Universal Sentence Encoder embeddings projected to 2D "
        "with PCA)"
    )
    ax.set_xlabel("principal component 1")
    ax.set_ylabel("principal component 2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Leave room on the right so the inline sentence labels are not clipped.
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min, x_max + 0.3 * (x_max - x_min))

    plt.tight_layout()
    plt.show()


def main():

    # A clean, well-separated set of supermarket aisles. Each aisle shares a
    # tight vocabulary so the topics form distinct clusters in the 2D PCA view
    # (mixed "bridge" sentences blur the clusters and hide the neighborhood).
    raw_sentences: list[str] = [
        # fruit
        "Apples are stacked in a pyramid on the shelf.",
        "Bananas hang in bunches near the scale.",
        "Strawberries are sold in small cartons.",
        "Oranges fill the wooden crate.",
        "Green grapes are priced per pound.",
        "A ripe mango sits on the stand.",
        "Pineapples lean against the coconuts.",
        "Peaches are arranged in a shallow tray.",
        # dairy
        "Fresh milk bottles fill the chiller.",
        "Cheddar cheese is cut at the deli counter.",
        "Yogurt tubs are next to the butter.",
        "Eggs are sold in cartons of twelve.",
        "Cream sits on the top shelf of the fridge.",
        "Butter is wrapped in foil.",
        "Soft cheese is kept in the chilled aisle.",
        "A block of cheese is on the middle shelf.",
        # bakery
        "Fresh bread is baked every morning.",
        "Bagels are stacked in the bakery aisle.",
        "A chocolate cake sits under the glass dome.",
        "Croissants are arranged in a basket.",
        "Sourdough loaves cool on the rack.",
        "Baguettes are wrapped in paper.",
        "Muffins are sold in packs of four.",
        "The bakery sells fresh rolls and pastries.",
        # beverages: deliberately avoids the words "cold" and "drinks", so the
        # query matches this aisle semantically rather than on exact words.
        "Soda cans are arranged on the middle shelf.",
        "Bottled water is sold by the case.",
        "Lemonade is stocked in the refrigerated aisle.",
        "Iced tea is displayed near the front of the store.",
        "Orange juice is kept in the cooler.",
        "Cola is on special this week.",
        "Sparkling water is sold in glass bottles.",
        "Ginger ale is stocked beside the soda.",
        # cleaning
        "Dish soap sits on the bottom shelf.",
        "Laundry detergent is heavy so it stays low.",
        "Paper towels are stacked on pallets.",
        "Bleach is kept away from the food aisles.",
        "Sponges and brushes fill the cleaning aisle.",
        "Glass cleaner is sold in spray bottles.",
        "Trash bags come in large rolls.",
        "Floor cleaner is priced per bottle.",
    ]

    raw_query: str = "Cold drinks to bring on a picnic."

    sentences = [preprocess(sentence) for sentence in raw_sentences]
    query = preprocess(raw_query)

    print(f"Embedding {len(sentences)} sentences and 1 query...")
    print(f'Query (preprocessed): "{query}"')
    sentence_embeddings = embed_text(sentences)
    query_embedding = embed_text([query])[0]

    graph = NSMWGraph(w=20, f=10, space=Space.COSINE, seed=42)
    for i, vector in enumerate(sentence_embeddings):
        graph.insert(Vertex(i, vector))

    print(f"Vertices: {graph.vertex_count()}")
    print(f"Edges: {graph.edge_count()}")

    k = 3
    results = graph.search(query_embedding, m=20, k=k)

    print(f'\nQuery: "{query}"')
    print(f"Top {k} nearest sentences:")
    for distance, vid, _ in results:
        print(f"    {sentences[vid]} (cosine distance: {distance:.4f})")

    visualize(graph, sentences, sentence_embeddings, query, query_embedding, results)


if __name__ == "__main__":
    main()
