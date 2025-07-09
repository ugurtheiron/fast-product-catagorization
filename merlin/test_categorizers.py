"""Example utilities and lightweight tests for Merlin."""

from __future__ import annotations

import json

from categorizer_top_k import (
    Category,
    categorize_product,
    load_category_file,
    ensure_category_index,
    get_top_k_paths,
    pick_best_category,
    guess_category,
)
import numpy as np
import pytest



def categorizer_single_test() -> None:
    from categorizer_single import CategoryClassifier
    import json

    # Initialize the classifier with the category list file
    classifier = CategoryClassifier(category_file="C:\\workspaces\\fast-product-catagorization\\Sporday_Kategori_Tüm_Sporlar.xlsx", 
                                    threshold_full = 0.6,
                                    threshold_partial = 0.5)

    # Single product classification example
    product_name = "Camo Hunting Shirt"
    result = classifier.classify(product_name)
    print("Product Name:", product_name)
    print("Classification Result:", result)

    # If you want the result in JSON (e.g., for programmatic use or API response)
    print("Result in JSON:", json.dumps(result, ensure_ascii=False))

    # Batch classification example (multiple products at once)
    product_list = ["Nike Air Zoom Running Shoes Men's", "Kids Basketball Shorts", "Unknown Product XYZ"]
    batch_results = classifier.classify_batch(product_list)
    for name, res in zip(product_list, batch_results):
        print(f"\nProduct Name: {name}")
        print("Result:", res)
# categorizer_single_test()


def categorizer_single_top_k_test():
    """Demonstrate the FAISS + GPT categorization helpers."""
    
    categories = load_category_file("C:\\workspaces\\fast-product-catagorization\\Sporday_Kategori_Tüm_Sporlar.xlsx")
    ensure_category_index(categories)
    product_list = ["Nike Air Zoom Running Shoes Men's", "Kids Basketball Shorts", "Unknown Product XYZ"]
    description = None
    for title in product_list:
        paths = get_top_k_paths(title, description, k=10)
        best = pick_best_category(f"{title}.", paths)
        print("Candidates:")
        for path in paths:
            print("-", path)
        print("Title:", title)
        print("Best:", best)
        # Or use the convenience wrapper
        # final_path = categorize_product(title, description, k=10)
        # print("Categorize product result:", final_path)
categorizer_single_top_k_test()
