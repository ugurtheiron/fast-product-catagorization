from openai import OpenAI
import numpy as np
import pickle
import os
from dotenv import load_dotenv
from utils import load_categories, preprocess_text

class CategoryClassifier:
    def __init__(self, category_file: str, embedding_model: str = "text-embedding-3-large",
                 cache_path: str = "category_embeddings.pkl", threshold_full: float = 0.8,
                 threshold_partial: float = 0.6):
        """
        Initialize the classifier by loading category data and embeddings.
        threshold_full: similarity threshold to accept full 3-level match.
        threshold_partial: if similarity is below threshold_full but above this, give higher-level category.
        """
        load_dotenv()  # Load environment variables from .env if present
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")
        self.client = OpenAI(api_key=api_key)

        self.embedding_model = embedding_model
        self.threshold_full = threshold_full
        self.threshold_partial = threshold_partial

        # Load category list
        self.categories = load_categories(category_file)  # using the utils.load_categories function
        self.num_categories = len(self.categories)
        # Try to load cached embeddings
        self.category_embeddings = None  # will hold a numpy array of shape (N, 1536)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                    # Ensure the cache corresponds to the current category list
                    if cache_data.get("categories") == self.categories:
                        self.category_embeddings = np.array(cache_data["embeddings"], dtype=np.float32)
            except Exception as e:
                print("Warning: Failed to load cache, will re-generate embeddings. Error:", e)
        # If no valid cache, generate embeddings
        if self.category_embeddings is None:
            self._generate_category_embeddings()
            # Save to cache
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump({"categories": self.categories, "embeddings": self.category_embeddings.tolist()}, f)
            except Exception as e:
                print("Warning: Could not save embeddings cache:", e)

    def _generate_category_embeddings(self, batch_size: int = 100):
        print("------------------------------------- EMBEDDINGS -------------------------------------")
        """Generate embeddings for all categories using OpenAI API."""
        embeddings = []
        # Process in batches to respect token limits and avoid timeouts
        for i in range(0, self.num_categories, batch_size):
            batch_texts = self.categories[i : i + batch_size]
            try:
                response = self.client.embeddings.create(model=self.embedding_model, input=batch_texts)
            except Exception as e:
                # Handle API errors gracefully
                raise RuntimeError(f"OpenAI API call failed for batch starting at index {i}: {e}")
            # OpenAI returns a list of embeddings in response.data
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        # Convert to numpy array for efficient math (and normalize for cosine similarity if desired)
        self.category_embeddings = np.array(embeddings, dtype=np.float32)
        # Optional: normalize embeddings to unit length to simplify cosine similarity calculation
        norms = np.linalg.norm(self.category_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10  # prevent division by zero
        self.category_embeddings = self.category_embeddings / norms
        # TODO: Consider using a vector database if self.num_categories becomes very large for faster search.


    def classify(self, product_name: str):
        print("------------------------------------- CLASSIFY -------------------------------------")
        """
        Classify a single product name into the best matching category path.
        Returns a dictionary with keys "Translation Level 1", "Translation Level 2", "Translation Level 3",
        or a message "not suitable for this category list" if no good match is found.
        """
        # Preprocess the input name
        clean_name = preprocess_text(product_name)
        if not clean_name:
            raise ValueError("Product name is empty after preprocessing.")
        # Get embedding for the product name
        try:
            response = self.client.embeddings.create(model=self.embedding_model, input=[clean_name])
        except Exception as e:
            raise RuntimeError(f"OpenAI API failed to embed product name: {e}")
        product_vector = np.array(response.data[0].embedding, dtype=np.float32)
        # Normalize the product vector
        norm = np.linalg.norm(product_vector)
        if norm == 0:
            # Edge case: zero vector (shouldn't happen for ada-002)
            norm = 1e-10
        product_vector = product_vector / norm
        # Compute cosine similarities with all category embeddings
        sims = np.dot(self.category_embeddings, product_vector)  # shape (num_categories,)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_category_path = self.categories[best_idx]
        print(f"score {best_score:.4f}")
        # Decide output based on similarity thresholds
        if best_score < self.threshold_partial:
            # No category is similar enough
            return "not suitable for this category list"
        elif best_score < self.threshold_full:
            # Moderate confidence: only return higher-level categories (e.g., Level 1 and 2)
            parts = best_category_path.split(" > ")
            if len(parts) >= 2:
                return {"Translation Level 1": parts[0], "Translation Level 2": parts[1], "Translation Level 3": None}
            else:
                # Only one level in category path
                return {"Translation Level 1": parts[0], "Translation Level 2": None, "Translation Level 3": None}
        else:
            # High confidence: return full category path (up to 3 levels)
            parts = best_category_path.split(" > ")
            # Pad the parts to length 3 with None if missing
            parts = parts + [None] * (3 - len(parts))
            return {"Translation Level 1": parts[0], "Translation Level 2": parts[1], "Translation Level 3": parts[2]}


    def classify_batch(self, product_names: list):
        print("------------------------------------- CLASSIFY BATCH -------------------------------------")
        """
        Classify a list of product names. Returns a list of results corresponding to each input.
        """
        # Preprocess all names
        clean_names = [preprocess_text(name) for name in product_names]
        # Remove any empty (post-cleaning) and handle accordingly
        # (For simplicity, we'll assume inputs are non-empty after cleaning; otherwise, we could put placeholder or error.)
        # Embed all names in one or multiple calls depending on list size
        embeddings = []
        batch_size = 100  # for example, embed 100 products per API call to avoid too large requests
        for i in range(0, len(clean_names), batch_size):
            batch = clean_names[i : i + batch_size]
            try:
                resp = self.client.embeddings.create(model=self.embedding_model, input=batch)
            except Exception as e:
                raise RuntimeError(f"OpenAI API failed for batch of products {i}-{i+len(batch)}: {e}")
            batch_embeds = [item.embedding for item in resp.data]
            embeddings.extend(batch_embeds)
        product_matrix = np.array(embeddings, dtype=np.float32)
        # Normalize each product embedding vector
        norms = np.linalg.norm(product_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        product_matrix = product_matrix / norms
        # Compute similarity matrix: shape = (num_products, num_categories)
        similarity_matrix = np.dot(product_matrix, self.category_embeddings.T)
        results = []
        for idx, sims in enumerate(similarity_matrix):
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            print(f"score {best_score:.4f}")
            best_path = self.categories[best_idx]
            if best_score < self.threshold_partial:
                result = "not suitable for this category list"
            elif best_score < self.threshold_full:
                parts = best_path.split(" > ")
                if len(parts) >= 2:
                    result = {"Translation Level 1": parts[0], "Translation Level 2": parts[1], "Translation Level 3": None}
                else:
                    result = {"Translation Level 1": parts[0], "Translation Level 2": None, "Translation Level 3": None}
            else:
                parts = best_path.split(" > ")
                parts = parts + [None] * (3 - len(parts))
                result = {"Translation Level 1": parts[0], "Translation Level 2": parts[1], "Translation Level 3": parts[2]}
            results.append(result)
        return results