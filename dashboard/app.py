import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Product Recommendation Engine",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ E-Commerce Recommendation Engine")
st.markdown("Hybrid ML system using SVD Collaborative Filtering + TF-IDF Content-Based filtering")

# ── Sidebar ────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Dashboard Stats",
    "🎯 Get Recommendations",
    "🔍 Similar Products",
    "📜 User History"
])

# ── Page 1: Stats ──────────────────────────────────────────
if page == "📊 Dashboard Stats":
    st.header("📊 Dataset & Model Overview")

    stats = requests.get(f"{API_URL}/stats").json()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users", f"{stats['total_users']:,}")
    col2.metric("Total Products", f"{stats['total_products']:,}")
    col3.metric("Total Reviews", f"{stats['total_reviews']:,}")
    col4.metric("Avg Rating", stats['avg_rating'])

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Models in use")
        for model in stats['models']:
            st.success(f"✅ {model}")
    with col2:
        st.subheader("Key metrics")
        st.info(f"🔵 Matrix Sparsity: {stats['sparsity']}")
        st.info("🔵 CF Weight: 70%")
        st.info("🔵 CB Weight: 30%")

    st.markdown("---")
    st.subheader("🏆 Most Popular Products")
    popular = requests.get(f"{API_URL}/popular?n=10").json()
    pop_df = pd.DataFrame(popular['popular'])
    pop_df.index += 1
    pop_df.columns = ['Product ID', 'Hybrid Score',
                      'Avg Rating', 'Popularity Score', 'Source']
    st.dataframe(pop_df[['Product ID', 'Avg Rating',
                          'Hybrid Score']], use_container_width=True)

# ── Page 2: Recommendations ────────────────────────────────
elif page == "🎯 Get Recommendations":
    st.header("🎯 Personalized Recommendations")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_id = st.text_input(
            "Enter User ID",
            value="A3OXHLG6DIBRW8",
            placeholder="e.g. A3OXHLG6DIBRW8"
        )
    with col2:
        n_recs = st.slider("Number of recommendations", 5, 20, 10)

    if st.button("🚀 Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            response = requests.get(
                f"{API_URL}/recommend",
                params={"user_id": user_id, "n": n_recs}
            )

        if response.status_code == 200:
            data = response.json()
            st.success(f"✅ Found {data['count']} recommendations "
                      f"(source: {data['source']})")

            recs_df = pd.DataFrame(data['recommendations'])
            recs_df.index += 1

            st.subheader("Top Recommendations")
            for i, row in recs_df.iterrows():
                with st.expander(
                    f"#{i} — {row['product_id']} "
                    f"(hybrid score: {row['hybrid_score']:.4f})"
                ):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Hybrid Score", row['hybrid_score'])
                    col2.metric("CF Score", row['cf_score'])
                    col3.metric("CB Score", row['cb_score'])
        else:
            st.error("Something went wrong. Is the API running?")

# ── Page 3: Similar Products ───────────────────────────────
elif page == "🔍 Similar Products":
    st.header("🔍 Find Similar Products")

    col1, col2 = st.columns([3, 1])
    with col1:
        product_id = st.text_input(
            "Enter Product ID",
            value="B002QWHJOU",
            placeholder="e.g. B002QWHJOU"
        )
    with col2:
        n_similar = st.slider("Number of similar products", 5, 20, 10)

    if st.button("🔍 Find Similar", type="primary"):
        with st.spinner("Finding similar products..."):
            response = requests.get(
                f"{API_URL}/similar/{product_id}",
                params={"n": n_similar}
            )

        if response.status_code == 200:
            data = response.json()
            st.success(f"✅ Found {len(data['similar'])} "
                      f"similar products (source: {data['source']})")

            sim_df = pd.DataFrame(data['similar'])
            sim_df.index += 1
            sim_df.columns = ['Product ID', 'Similarity Score',
                              'Avg Rating', 'Review Count']
            st.dataframe(sim_df, use_container_width=True)

            st.subheader("Similarity Score Distribution")
            st.bar_chart(sim_df.set_index('Product ID')['Similarity Score'])
        else:
            st.error(f"Product {product_id} not found.")

# ── Page 4: User History ───────────────────────────────────
elif page == "📜 User History":
    st.header("📜 User Review History")

    user_id = st.text_input(
        "Enter User ID",
        value="A3OXHLG6DIBRW8",
        placeholder="e.g. A3OXHLG6DIBRW8"
    )

    if st.button("📜 Get History", type="primary"):
        with st.spinner("Loading history..."):
            response = requests.get(f"{API_URL}/user/{user_id}/history")

        if response.status_code == 200:
            data = response.json()

            col1, col2 = st.columns(2)
            col1.metric("Total Reviews Written", data['total_reviews'])
            col2.metric("Average Rating Given", data['avg_rating_given'])

            st.subheader("Review History (Top 20 by rating)")
            hist_df = pd.DataFrame(data['history'])
            hist_df.index += 1
            st.dataframe(hist_df, use_container_width=True)
        else:
            st.error(f"User {user_id} not found.")