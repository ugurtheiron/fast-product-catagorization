# Merlin Product Categorization

Merlin maps e-commerce product titles to a three-level category tree. It uses a two step approach:

1. **Nearest-neighbour search** – Each category path is embedded once and stored in a FAISS index. A product title (optionally combined with its description) is embedded the same way and the closest category paths are retrieved.
2. **GPT selection** – The shortlist is presented to GPT which picks the final category. This keeps the number of model calls low while still leveraging GPT's reasoning ability.

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
from merlin import load_category_file, ensure_category_index, guess_category

categories = load_category_file("Sporday_Kategori_Tüm_Sporlar.xlsx")
ensure_category_index(categories)
guess = guess_category("Camo Hunting Shirt", None)
print(guess)
```
