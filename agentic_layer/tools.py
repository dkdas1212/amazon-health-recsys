"""
tools.py
Wraps the EXISTING recsys FastAPI endpoints (main.py) as LangChain tools.
Matches the real routes: /recommend, /similar/{product_id}, /popular, /user/{user_id}/history, /search
"""

import os
import requests
from langchain_core.tools import tool

BASE_URL = os.getenv("RECSYS_API_URL", "http://localhost:8000")


@tool
def hybrid_recommend(user_id: str, n: int = 10) -> dict:
    """Get hybrid (SVD + content-based) recommendations for a user_id.
    Falls back to popular products automatically if the user is unknown.
    Use this as the default recommendation tool."""
    r = requests.get(f"{BASE_URL}/recommend", params={"user_id": user_id, "n": n}, timeout=10)
    r.raise_for_status()
    return r.json()


@tool
def similar_products(product_id: str, n: int = 10) -> dict:
    """Get content-based recommendations similar to a given product_id.
    Use this for 'similar to X product' style queries."""
    r = requests.get(f"{BASE_URL}/similar/{product_id}", params={"n": n}, timeout=10)
    r.raise_for_status()
    return r.json()


@tool
def popular_products(n: int = 10) -> dict:
    """Get globally popular products (by rating + review count).
    Use this for 'what's trending' or as a cold-start fallback."""
    r = requests.get(f"{BASE_URL}/popular", params={"n": n}, timeout=10)
    r.raise_for_status()
    return r.json()


@tool
def user_history(user_id: str) -> dict:
    """Get a user's past reviewed/purchased products and ratings.
    Use this to answer questions about what the user has already bought or rated."""
    r = requests.get(f"{BASE_URL}/user/{user_id}/history", timeout=10)
    r.raise_for_status()
    return r.json()


@tool
def category_search(keywords: str, n: int = 10) -> dict:
    """Search the FULL product catalog (not just top recommendations) for products
    whose reviews mention any of the given comma-separated keywords, ranked by rating.
    Use this for category-style requests like coffee, tea, trail mix, spicy snacks, pet items, etc.
    Example keywords: 'coffee,tea,espresso'"""
    r = requests.get(f"{BASE_URL}/search", params={"keywords": keywords, "n": n}, timeout=10)
    r.raise_for_status()
    return r.json()


ALL_TOOLS = [hybrid_recommend, similar_products, popular_products, user_history, category_search]