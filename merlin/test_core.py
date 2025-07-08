"""Example utilities and lightweight tests for Merlin."""

from __future__ import annotations

import json

from merlin.categorizer import (
    Category,
    categorize_product,
    ensure_category_index,
    get_top_k_paths,
    pick_best_category,
    guess_category,
)
import numpy as np
import pytest



def demo_old_classifier(category_file: str) -> None:
    """Run the original classifier on a couple of products."""
    from merlin.core import CategoryClassifier

    classifier = CategoryClassifier(
        category_file=category_file,
        threshold_full=0.6,
        threshold_partial=0.5,
    )

    product_name = "Camo Hunting Shirt"
    result = classifier.classify(product_name)
    print("Product Name:", product_name)
    print("Classification Result:", result)
    print("Result in JSON:", json.dumps(result, ensure_ascii=False))

    product_list = [
        "Nike Air Zoom Running Shoes Men's",
        "Kids Basketball Shorts",
        "Unknown Product XYZ",
    ]
    batch_results = classifier.classify_batch(product_list)
    for name, res in zip(product_list, batch_results):
        print(f"\nProduct Name: {name}")
        print("Result:", res)


def demo_new_pipeline(categories: list[Category]) -> None:
    """Demonstrate the FAISS + GPT categorization helpers."""

    ensure_category_index(categories)
    title = "Camo Hunting Shirt"
    description = None
    paths = get_top_k_paths(title, description, k=10)
    best = pick_best_category(f"{title}.", paths)
    print("Candidates:", paths)
    print("Best:", best)
    # Or use the convenience wrapper
    final_path = categorize_product(title, description, k=10)
    print("Categorize product result:", final_path)


def test_category_dataclass():
    c = Category(path="a > b")
    assert c.path == "a > b"


def test_new_pipeline(monkeypatch, tmp_path):
    """Verify knn lookup and GPT picker using fake OpenAI responses."""
    faiss = pytest.importorskip("faiss")

    # ------------------------------------------------------------------
    # Build a small FAISS index with three dummy categories
    # ------------------------------------------------------------------
    dims = 3
    arr = np.eye(dims, dtype="float32")
    faiss.normalize_L2(arr)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dims))
    ids = np.arange(dims).astype("int64")
    index.add_with_ids(arr, ids)
    index_path = tmp_path / "cat.faiss"
    faiss.write_index(index, str(index_path))
    with open(str(index_path) + ".json", "w", encoding="utf-8") as f:
        json.dump(["A", "B", "C"], f)

    # ------------------------------------------------------------------
    # Fake OpenAI client returning deterministic embeddings
    # ------------------------------------------------------------------
    class FakeEmbeddings:
        def create(self, *, input, **kwargs):
            data = []
            for text in input:
                if text == "A":
                    vec = [1.0, 0.0, 0.0]
                elif text == "B" or text == "something":
                    vec = [0.0, 1.0, 0.0]
                else:
                    vec = [0.0, 0.0, 1.0]
                data.append(type("D", (), {"embedding": vec})())
            return type("Resp", (), {"data": data})

    class FakeClient:
        def __init__(self, *_, **__):
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr("merlin.categorizer.OpenAI", FakeClient)

    # ------------------------------------------------------------------
    # Run lookup and picker
    # ------------------------------------------------------------------
    paths = get_top_k_paths(
        "something", None, k=2, dims=dims, index_path=str(index_path)
    )
    assert paths == ["B", "A"]

    best = pick_best_category("something", paths)
    assert best == "B"

    # Convenience single guess helper
    guess = guess_category("something", None, dims=dims, index_path=str(index_path))
    assert guess == "B"
