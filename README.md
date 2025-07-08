# Merlin Product Categorization

Merlin maps e-commerce product titles to a three-level category tree using a
combination of vector search and a lightweight GPT prompt. Each category path is
embedded once and stored in a FAISS index. When classifying a product, the
nearest category paths are looked up and the final choice is made by GPT from
that short list.

## Features
- Reusable FAISS index for category embeddings
- Single or batch product classification
- `.env` based API configuration
- Optional CLI via `merlin-classify "Product name"`

## Quick start
```bash
# Install dependencies
pip install -e .

# Provide your OpenAI key
echo 'OPENAI_API_KEY="sk-..."' > .env

# Run the example
python example/quickstart.py
```

Use the helpers directly from Python:

```python
from merlin import load_category_file, ensure_category_index, categorize_product

categories = load_category_file("Sporday_Kategori_Tüm_Sporlar.xlsx")
ensure_category_index(categories)
result = categorize_product("Camo Hunting Shirt", None)
print(result)
```
