
Merlin classifies e-commerce product names into a **3-level hierarchy** using
OpenAI’s `text-embedding-ada-002` vectors and fast cosine-similarity search.

## Features
* **One-time embedding cache** for thousands of categories  
* Handles **single or batch** product names  
* Configurable confidence thresholds  
* `.env`-based secrets (no keys in code)  
* Optional CLI → `merlin-classify "Nike Running Shoes"`

├── merlin/        # library code
├── tests/         # pytest suite
├── examples/      # usage demos

## Quick Start

```bash
# 1 – Install
pip install -e .

# 2 – Add your OpenAI key
echo 'OPENAI_API_KEY="sk-…"' > .env

# 3 – Classify a product
python examples/quickstart.py
# or
merlin-classify "Camo Hunting Shirt"