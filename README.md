# Merlin Product Categorization

Merlin helps automatically map product titles to a three-level category tree using AI. I built this because manually categorizing products is painfully slow, and I needed something that could handle both speed and accuracy depending on the situation.

## How it works

There are two ways to categorize products, which one you choose depends on what you need:

### The Fast Way (Single Similarity)
This is my go-to for bulk processing. Here's what happens:
- First, we create embeddings for all your categories (this happens once and gets cached)
- When you throw a product name at it, we embed that too
- Then it's just math - find the category with the highest cosine similarity
- Based on confidence thresholds, you get either the full category path, a partial match, or "nope, doesn't fit"

this approach:
- Lightning fast once set up
- No extra API calls = lower costs
- Predictable results every time
- Perfect when you need to process thousands of products

### The Accurate Way (Top-K + GPT)
When accuracy matters more than speed, this is the way to go:
- Same embedding setup, but we store everything in a FAISS index for faster searching
- For each product, we find the 15 most similar categories using vector search
- Here's the magic: we send those 15 candidates along with the product to GPT and let it decide
- GPT understands context way better than pure similarity matching

This works better because:
- GPT catches nuances that similarity scores miss
- Handles edge cases beautifully
- More accurate overall, especially for tricky products
- Worth the extra API cost for high-value items

## What's included
- Two different classification approaches (pick your poison!)
- Embeddings get cached so you don't waste time/money regenerating them
- Works with single products or batch processing
- You can tweak the similarity thresholds to be more or less picky
- Just drop your OpenAI key in a `.env` file and you're good to go
- Bonus: there's even a CLI if you want to test things quickly

## Getting started
```bash
# Install the thing
pip install -e .

# Add your OpenAI key (you know the drill)
echo 'OPENAI_API_KEY="sk-your-key-here"' > .env

# Try it out with the example
python merlin/test_categorizers.py
```

Use the helpers directly from Python:

```python
# Method 1: Single Similarity (Fast)
from merlin.categorizer_single import CategoryClassifier

classifier = CategoryClassifier("Sporday_Kategori_Tüm_Sporlar.xlsx")
result = classifier.classify("Camo Hunting Shirt")
print(result)

# Method 2: Top-K + GPT Selection (Accurate)
from merlin.categorizer_top_k import load_category_file, ensure_category_index, categorize_product

categories = load_category_file("Sporday_Kategori_Tüm_Sporlar.xlsx")
ensure_category_index(categories)
result = categorize_product("Camo Hunting Shirt", None)
print(result)
```
