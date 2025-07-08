"""Helpers for category search and GPT-based selection."""

from __future__ import annotations

import json
import os

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from openai import OpenAI

from utils import load_categories, preprocess_text

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore


@dataclass
class Category:
    """Simple category node with a full path string."""

    path: str


def load_category_file(file_path: str) -> List[Category]:
    """Return Category objects loaded from an Excel file."""
    paths = load_categories(file_path)
    return [Category(path=p) for p in paths]


def _require_faiss() -> None:
    """Ensure that faiss is installed before continuing."""
    if faiss is None:
        raise ImportError("faiss is required for this operation")


def ensure_category_index(
    categories: List[Category],
    dims: int = 256,
    index_path: str = "cache/cat.faiss",
) -> "faiss.Index":
    """Ensure a FAISS index exists for the given categories."""
    # We embed each category path and store the normalized vectors in a
    # FAISS index on disk. Subsequent runs reuse the saved index to avoid
    # expensive embedding calls.
    _require_faiss()
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    map_path = index_path + ".json"
    if os.path.exists(index_path) and os.path.exists(map_path):
        return faiss.read_index(index_path)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    texts = [c.path for c in categories]
    embeddings: List[List[float]] = []
    batch = 100
    for i in range(0, len(texts), batch):
        # Request embeddings in manageable batches to respect API limits.
        resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=texts[i : i + batch],
            dimensions=dims,
        )
        embeddings.extend([d.embedding for d in resp.data])

    # Normalise the vectors to unit length before indexing
    arr = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(arr)

    # Build a simple Inner Product index mapping each vector by position
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dims))
    ids = np.arange(len(arr)).astype("int64")
    index.add_with_ids(arr, ids)

    faiss.write_index(index, index_path)
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump([c.path for c in categories], f, ensure_ascii=False)
    return index


def _load_paths(index_path: str) -> List[str]:
    """Read the path mapping created alongside the FAISS index."""
    map_path = index_path + ".json"
    with open(map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_top_k_paths(
    title: str,
    description: Optional[str],
    k: int = 15,
    dims: int = 256,
    index_path: str = "cache/cat.faiss",
) -> List[str]:
    """Return the paths of the *k* nearest categories to the product text."""
    _require_faiss()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Combine title and optional description then preprocess the text
    query = f"{title}. {description or ''}".strip()
    query = preprocess_text(query)
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=[query],
        dimensions=dims,
    )
    vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)

    # Normalise and query the index for the nearest neighbours
    faiss.normalize_L2(vec)
    index = faiss.read_index(index_path)
    _dists, idxs = index.search(vec, k)

    # Resolve integer ids back to category paths
    paths = _load_paths(index_path)
    return [paths[i] for i in idxs[0]]


def pick_best_category(
    product_text: str,
    candidates: List[str],
    *,
    temperature: float = 0.2,
    model: str = "gpt-4o-mini",
) -> Optional[str]:
    """Return the single best category using a GPT function call."""

    if not candidates:
        return None

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system = "You are a category picker. Return exactly one full path from the list or 'NONE'."
    bullet_list = "\n".join(f"- {c}" for c in candidates)
    user = f"Product: {product_text}\n\nCandidates:\n{bullet_list}"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "pick_category",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "pick_category"}},
    )

    args = json.loads(
        resp.choices[0].message.tool_calls[0].function.arguments
    )
    path = args["path"].strip()
    if path.upper() == "NONE":
        return None
    return path


def categorize_product(
    title: str,
    description: Optional[str],
    k: int = 15,
    dims: int = 256,
) -> Optional[str]:
    """High-level helper that finds candidates then selects the best one."""
    product_text = f"{title}. {description or ''}".strip()
    candidates = get_top_k_paths(title, description, k=k, dims=dims)
    return pick_best_category(product_text, candidates)


def guess_category(
    title: str,
    description: Optional[str],
    dims: int = 256,
    index_path: str = "cache/cat.faiss",
) -> str:
    """Return the single best category path using k-NN only."""

    return get_top_k_paths(
        title, description, k=1, dims=dims, index_path=index_path
    )[0]
