import pandas as pd
import os
import re
import random

def load_categories(file_path: str):
    print("------------------------------------- DATA PREPARATION -------------------------------------")
    """Load category hierarchy from an Excel file and return a list of unique category path strings."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Category file not found: {file_path}")
    
    # Only read necessary columns to save memory
    df = pd.read_excel(file_path, usecols=["Translation Level 1", "Translation Level 2", "Translation Level 3"], dtype=str)
    df.fillna("", inplace=True)  # replace NaN with empty string for combination

    # Combine levels into a "Level1 > Level2 > Level3" string
    def combine_levels(row):
        # Only join non-empty parts
        parts = [row["Translation Level 1"].strip()]
        if row["Translation Level 2"].strip():
            parts.append(row["Translation Level 2"].strip())
        if row["Translation Level 3"].strip():
            parts.append(row["Translation Level 3"].strip())
        return " > ".join(parts)
    # Apply combination and drop duplicates
    df["Category_Path"] = df.apply(combine_levels, axis=1)
    unique_paths = df["Category_Path"].drop_duplicates().tolist()
    print_overall_stats(df)
    print_random_paths(df, n=10)
    return unique_paths



def preprocess_text(text: str) -> str:
    """Lowercase the text and remove special characters for cleaner embedding."""
    text = text.lower()
    # Remove any character that is not a letter, digit, or whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text).strip()
    return text

def print_overall_stats(df: pd.DataFrame) -> None:
    print("Catalogue summary")
    print("-" * 40)
    print(f"Unique Level-1 categories : {df['Translation Level 1'].nunique()}")
    print(f"Unique Level-2 subcats    : {df['Translation Level 2'].replace('', pd.NA).dropna().nunique()}")
    print(f"Unique Level-3 subsubcats : {df['Translation Level 3'].replace('', pd.NA).dropna().nunique()}")
    print(f"Total paths      : {df['Category_Path'].nunique()}")
    print("-" * 40)


def print_random_paths(df: pd.DataFrame, n: int = 10) -> None:
    """
    Print n random unique category paths for sanity-checking.
    """
    paths = df["Category_Path"].unique().tolist()
    print(f"\n🎲 {n} random category paths:")
    for p in random.sample(paths, min(n, len(paths))):
        print("  -", p)