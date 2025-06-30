from core import CategoryClassifier
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