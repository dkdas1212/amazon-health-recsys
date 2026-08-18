"""
agent.py
LangGraph agent that sits in front of the existing recsys FastAPI service.

Flow:
    parse_intent -> select_tool(s) -> apply_constraints -> format_response

The LLM (Groq) only handles intent parsing, tool selection, constraint
checking, and the final natural-language explanation. All actual
recommendation math stays in your existing SVD/TF-IDF/hybrid models.
"""

import os
import json
from typing import TypedDict, Optional

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from tools import ALL_TOOLS

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))
llm_with_tools = llm.bind_tools(ALL_TOOLS)
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


class AgentState(TypedDict):
    query: str
    user_id: Optional[str]
    intent: Optional[dict]
    tool_results: Optional[list]
    constraints_ok: Optional[bool]
    category_match_found: Optional[bool]
    response: Optional[str]
    trace: list  # decision log for explainability / debugging


def parse_intent(state: AgentState) -> AgentState:
    """Extract structured intent (goal, filters like budget/allergen/category) from the query."""
    prompt = (
        "Extract the user's shopping intent as JSON with keys: "
        "goal (string), budget_max (number or null), exclude_terms (list of strings), "
        "category (string or null), reference_item (string or null). "
        f"Query: {state['query']}"
    )
    result = llm.invoke(prompt)
    try:
        intent = json.loads(result.content)
    except json.JSONDecodeError:
        intent = {"goal": state["query"], "budget_max": None, "exclude_terms": [], "category": None, "reference_item": None}
    state["intent"] = intent
    state["trace"] = state.get("trace", []) + [f"Parsed intent: {intent}"]
    return state


def select_and_call_tools(state: AgentState) -> AgentState:
    """Let the LLM decide which tool(s) to call based on parsed intent, then execute them."""
    intent = state["intent"]
    context = f"User intent: {intent}. User id: {state.get('user_id')}."
    ai_msg = llm_with_tools.invoke(context)

    results = []
    for call in getattr(ai_msg, "tool_calls", []) or []:
        tool_fn = TOOLS_BY_NAME.get(call["name"])
        if tool_fn:
            try:
                out = tool_fn.invoke(call["args"])
                results.append({"tool": call["name"], "args": call["args"], "result": out})
            except Exception as e:
                results.append({"tool": call["name"], "args": call["args"], "error": str(e)})

    # Fallback: if no tool produced usable product recommendations
    # (either no tools were called, or the ones called errored/returned nothing),
    # default to hybrid_recommend so the user always gets something.
    def _has_products(results):
        for entry in results:
            result = entry.get("result")
            if entry.get("error"):
                continue
            items = None
            if isinstance(result, dict):
                for key in ("recommendations", "similar", "popular", "results"):
                    if key in result:
                        items = result[key]
                        break
            elif isinstance(result, list):
                items = result
            if items:
                return True
        return False

    if not _has_products(results) and state.get("user_id"):
        out = TOOLS_BY_NAME["hybrid_recommend"].invoke({"user_id": state["user_id"], "n": 10})
        results.append({"tool": "hybrid_recommend", "args": {"user_id": state["user_id"]}, "result": out})

    state["tool_results"] = results
    state["trace"] = state["trace"] + [f"Called tools: {[r['tool'] for r in results]}"]
    return state


def search_category_if_relevant(state: AgentState) -> AgentState:
    """If the query mentions a known category, search the FULL catalog for it
    via the /search endpoint (not just the narrow top-N already fetched).
    This replaces the old approach of filtering an already-small
    recommendation list, which could miss real matches elsewhere in the data."""
    intent = state["intent"]
    goal_text = (intent.get("goal") or state["query"]).lower()

    # Each group: broad "triggers" (words the user might actually type) mapped to
    # specific "search_terms" (what to look for in review text). This lets generic
    # words like "drinks"/"snacks"/"food" route to a real search instead of falling
    # through to the default top-5 recommendations every time.
    keyword_groups = {
        "coffee_tea": {
            "triggers": ["coffee", "tea", "k-cup", "kcup", "keurig", "espresso"],
            "search_terms": ["coffee", "tea", "k-cup", "kcup", "keurig", "espresso"],
        },
        "trail_mix_bars": {
            "triggers": ["trail mix", "granola", "muesli", "bar", "protein bar"],
            "search_terms": ["trail mix", "granola", "muesli", "bar", "protein bar"],
        },
        "spicy_savory": {
            "triggers": ["jalapeno", "spicy", "salty", "savory", "chips"],
            "search_terms": ["jalapeno", "spicy", "salty", "savory", "chips"],
        },
        "sweet_treat": {
            "triggers": ["sweet", "sugar", "treat", "candy", "chocolate", "dessert"],
            "search_terms": ["sweet", "sugar", "treat", "candy", "chocolate", "dessert"],
        },
        "pet": {
            "triggers": ["dog", "dogs", "pet", "cat", "cats", "furry"],
            "search_terms": ["dog", "dogs", "pet", "cat", "cats", "furry"],
        },
        "breakfast": {
            "triggers": ["morning", "breakfast", "wake up"],
            "search_terms": ["morning", "breakfast", "wake up"],
        },
        "lunch": {
            "triggers": ["lunch", "midday", "filling"],
            "search_terms": ["lunch", "midday", "filling"],
        },
        "dinner": {
            "triggers": ["dinner", "evening", "hearty"],
            "search_terms": ["dinner", "evening", "hearty"],
        },
        # Generic buckets: broad words the user is likely to actually type,
        # mapped to a union of specific terms to search for in review text.
        "drink_generic": {
            "triggers": ["drink", "drinks", "beverage", "beverages"],
            "search_terms": ["coffee", "tea", "k-cup", "kcup", "keurig", "espresso",
                              "soda", "juice", "drink", "beverage"],
        },
        "snack_generic": {
            "triggers": ["snack", "snacks"],
            "search_terms": ["trail mix", "granola", "muesli", "bar", "jalapeno",
                              "spicy", "salty", "savory", "chips", "sweet", "sugar",
                              "treat", "candy", "chocolate"],
        },
        "healthy_generic": {
            "triggers": ["healthy", "nutritious", "wholesome", "clean eating"],
            "search_terms": ["healthy", "natural", "organic", "vitamin", "protein",
                              "nutritious", "wholesome", "no sugar", "low calorie"],
        },
    }

    matched_group = None
    for group, spec in keyword_groups.items():
        if any(kw in goal_text for kw in spec["triggers"]):
            matched_group = group
            break

    if not matched_group:
        state["category_match_found"] = None
        state["trace"] = state["trace"] + ["No category keyword detected in query — skipping category search"]
        return state

    search_terms = keyword_groups[matched_group]["search_terms"]
    out = TOOLS_BY_NAME["category_search"].invoke({"keywords": ",".join(search_terms), "n": 10})
    results = out.get("results", [])

    if results:
        state["tool_results"] = [{"tool": "category_search", "args": {"keywords": search_terms}, "result": out}]
        state["category_match_found"] = True
        state["trace"] = state["trace"] + [f"Searched full catalog for '{matched_group}' — found {len(results)} real matches"]
    else:
        state["tool_results"] = []
        state["category_match_found"] = False
        state["trace"] = state["trace"] + [f"Searched full catalog for '{matched_group}' — NO matches found anywhere in the data"]

    return state


def apply_constraints(state: AgentState) -> AgentState:
    """Filter tool results by minimum rating/score. Your current API doesn't return
    price or product name/category fields, so budget/allergen filtering isn't possible
    yet — this only enforces a minimum quality bar (avg_rating or score >= 3.0)."""
    filtered = []
    for entry in state["tool_results"]:
        result = entry.get("result")
        items = None
        if isinstance(result, dict):
            # your API nests the list under keys like "recommendations", "similar", "popular"
            for key in ("recommendations", "similar", "popular", "results"):
                if key in result:
                    items = result[key]
                    break
        elif isinstance(result, list):
            items = result

        if isinstance(items, list):
            kept = [
                item for item in items
                if not isinstance(item, dict)
                or item.get("hybrid_score", item.get("avg_rating", 5)) >= 3.0
            ]
            filtered.append({**entry, "result": kept})
        else:
            filtered.append(entry)

    state["tool_results"] = filtered
    state["constraints_ok"] = True
    state["trace"] = state["trace"] + ["Applied minimum-rating quality filter (no price/allergen fields available yet)"]
    return state


def format_response(state: AgentState) -> AgentState:
    """Turn filtered tool results into a natural-language, explainable answer."""
    # Extract the real product_ids that actually came back from the tools,
    # so the LLM can't invent fake ones.
    real_ids = []
    for entry in state["tool_results"]:
        result = entry.get("result")
        items = None
        if isinstance(result, dict):
            for key in ("recommendations", "similar", "popular", "results"):
                if key in result:
                    items = result[key]
                    break
        elif isinstance(result, list):
            items = result
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and "product_id" in item:
                    real_ids.append(item["product_id"])

    category_note = ""
    if state.get("category_match_found") is False:
        category_note = (
            "IMPORTANT: a full catalog search found NO products anywhere whose reviews "
            "mention the requested category — do not invent or imply any exist. Say clearly "
            "that no matching products were found in the catalog for that category, and do "
            "not show unrelated 'general' recommendations instead unless the user asks for that. "
        )
    elif state.get("category_match_found") is True:
        category_note = (
            "These results come from a real search across the full product catalog for the "
            "requested category (based on review text mentioning it), not just a small "
            "recommendation list. Mention this was found via review-text search, so it's "
            "still approximate, not a guaranteed structured category. "
        )

    prompt = (
        f"User asked: {state['query']}\n"
        f"Tool results after constraint filtering: {json.dumps(state['tool_results'])[:4000]}\n"
        f"VALID product_id values (use ONLY these, exactly as written, never invent or reformat "
        f"a product_id): {real_ids}\n"
        "Write a short, friendly answer listing the top recommendations by their exact product_id "
        "from the VALID list above, and briefly explain why each was chosen using ONLY the fields "
        "present in the data (hybrid_score, avg_rating, review_count, similarity_score). "
        "IMPORTANT: the data has no price or ingredient/allergen fields. "
        "Do NOT claim you filtered by price, budget, or allergens/nuts — if the user asked for "
        "that, explicitly say this data isn't available yet instead of pretending it was applied. "
        "IMPORTANT: this catalog is Amazon Health & Personal Care products (supplements, "
        "snacks, drinks, pet items, personal care) — it does NOT contain full 'meals' or "
        "recipes. If the user asked for 'meals' or something this catalog likely doesn't "
        "have, say so plainly rather than presenting unrelated top-rated items as if they "
        "matched the request. "
        f"{category_note}"
        "If VALID product_id list is empty, say no recommendations were found instead of making any up."
    )
    result = llm.invoke(prompt)
    state["response"] = result.content
    state["trace"] = state["trace"] + ["Formatted final response"]
    return state


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("parse_intent", parse_intent)
    graph.add_node("select_tool", select_and_call_tools)
    graph.add_node("search_category", search_category_if_relevant)
    graph.add_node("apply_constraints", apply_constraints)
    graph.add_node("format_response", format_response)

    graph.set_entry_point("parse_intent")
    graph.add_edge("parse_intent", "select_tool")
    graph.add_edge("select_tool", "search_category")
    graph.add_edge("search_category", "apply_constraints")
    graph.add_edge("apply_constraints", "format_response")
    graph.add_edge("format_response", END)
    return graph.compile()


AGENT = build_graph()

if __name__ == "__main__":
    out = AGENT.invoke({
        "query": "find me healthy snacks under $15 similar to what I bought last month, but avoid nuts",
        "user_id": "123",
        "trace": [],
    })
    print(out["response"])
    print("\n--- trace ---")
    for step in out["trace"]:
        print("-", step)