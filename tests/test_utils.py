"""Offline tests for merlin.utils (no OpenAI calls, no network)."""

import pandas as pd
import pytest

from merlin.utils import load_categories, preprocess_text

LEVELS = ["Translation Level 1", "Translation Level 2", "Translation Level 3"]


def _write_catalogue(path, rows):
    pd.DataFrame(rows, columns=LEVELS).to_excel(path, index=False)
    return str(path)


def test_preprocess_text_lowercases_and_strips_punctuation():
    # The apostrophe becomes a separator, so "Men's" turns into two tokens.
    assert preprocess_text("Nike Air Zoom Running Shoes Men's") == "nike air zoom running shoes men s"


def test_preprocess_text_collapses_whitespace():
    assert preprocess_text("  Camo\t Hunting \n Shirt  ") == "camo hunting shirt"


def test_preprocess_text_keeps_digits_and_empties_symbol_only_input():
    assert preprocess_text("iPhone 15 Pro-Max") == "iphone 15 pro max"
    assert preprocess_text("!!! --- ???") == ""


def test_load_categories_drops_duplicate_paths(tmp_path):
    xlsx = _write_catalogue(
        tmp_path / "categories.xlsx",
        [
            ["Sports", "Running", "Shoes"],
            ["Sports", "Running", "Shoes"],
            ["Sports", "Running", "Shorts"],
        ],
    )

    assert load_categories(xlsx) == [
        "Sports > Running > Shoes",
        "Sports > Running > Shorts",
    ]


def test_load_categories_skips_empty_levels_and_strips_whitespace(tmp_path):
    xlsx = _write_catalogue(
        tmp_path / "categories.xlsx",
        [
            [" Sports ", " Running ", None],
            ["Outdoor", None, None],
            ["Outdoor", "", "Tents"],
        ],
    )

    # A missing level is left out of the path; leaf levels keep their parent chain.
    assert load_categories(xlsx) == [
        "Sports > Running",
        "Outdoor",
        "Outdoor > Tents",
    ]


def test_load_categories_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_categories(str(tmp_path / "nope.xlsx"))
