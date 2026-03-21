import streamlit as st

# -------------------------------------Mongo DB------------------------------------------
from local_mongo import get_collection, fetch_collation, insert_data


# Cache collection
@st.cache_resource
def get_cached_collection(db_name, collection_name):
    return get_collection(db_name, collection_name)


# -------------------------------------------------------------------------------------

# --- Page Configuration ---
st.set_page_config(
    page_title="Streamlit RAG Evaluator",
    page_icon=":material/monitoring:",
    layout="wide",
)

st.title(":material/monitoring: RAG Evaluation")
st.markdown(
    "This dashboard presents a collection of highly interactive ECharts visualizations integrated into Streamlit apps"
    ",designed to help benchmark RAG techniques and uncover the most accurate and actionable insights.  \n"
)

# --- Value Holder ---
config = {}

if "dataframe" not in st.session_state:
    st.session_state.dataframe = None


# --- SIDEBAR: Filters + Info ---
with st.sidebar:
    st.title(":material/filter_alt: Filters")

    # Collection Name
    collection_name = st.text_input("Collection Name", value="rag_eval_2")

    # Chunking Type
    chunking_type = st.selectbox("Chunking Type", ["Default Chunking", "LLM Chunking"])

    # Conditional UI
    if chunking_type == "Default Chunking":
        config["k"] = st.number_input("K Value", min_value=1, value=3)
        config["size"] = st.number_input("Chunk Size", min_value=1, value=8)

    elif chunking_type == "LLM Chunking":
        config["llm_model"] = st.selectbox("LLM Model", ["gpt", "falcon3"])
        config["temperature"] = st.slider("Temperature", 0.0, 1.0, 0.2)

    # Embedding Model
    embedding_model = st.selectbox(
        "Embedding Model",
        ["all-MiniLM-L6-v2", "text-embedding-3-small", "text-embedding-3-large"],
    )

    # RAG Preprocess
    rag_preporcess = st.multiselect("Preprocess", options=["Re-Rank", "Re-Query"])

    # Mongo Collection
    if collection_name:
        collection = get_cached_collection("rag_db", collection_name)
        st.session_state.dataframe = fetch_collation(collection)

    # Submit
    if st.button("Submit"):
        if not collection_name.strip():
            st.error("Collection Name is required")
        else:
            st.session_state.document = {
                "collection_name": collection_name,
                "chunking": {"type": chunking_type, "config": config},
                "embedding_model": embedding_model,
                "rag_preprocess": rag_preporcess,
            }
            # insert_data(collection, st.session_state.document)


col1, col2, col3, col4 = st.columns(4)

with col1:

    mrr = st.session_state.dataframe["eval_default_mrr"]

    current = mrr.iloc[0]
    top_value = mrr.max()

    delta = current - top_value

    st.metric(
        label="MRR (Default)",
        value=f"{current:.4f}",
        delta=f"{delta:+.4f}",
        border=True,
        chart_data=mrr,
        chart_type="line",
    )

with col2:

    llm_mrr_series = st.session_state.dataframe["eval_llm_mrr"]
    current = llm_mrr_series.iloc[0]
    previous = llm_mrr_series.iloc[1]

    delta = current - previous

    st.metric(
        label="LLM MRR",
        value=f"{current:.2f}",
        delta=f"{delta:+.2f}",
        border=True,
        chart_data=llm_mrr_series,
        chart_type="area",
    )

with col3:

    delta_mrr = (
        st.session_state.dataframe["eval_llm_mrr"]
        - st.session_state.dataframe["eval_default_mrr"]
    )

    st.metric(
        label="Δ MRR (LLM vs Default)",
        value=f"{delta_mrr.iloc[0]:+.4f}",
        delta=f"{delta_mrr.iloc[0] - delta_mrr.iloc[1]:+.4f}",
        border=True,
        chart_data=delta_mrr,
        chart_type="line",
    )

with col4:

    llm_mrr_series = st.session_state.dataframe["eval_llm_mrr"]
    best = llm_mrr_series.max()
    st.metric(
        label="Best LLM MRR",
        value=f"{best:.4f}",
        border=True,
        chart_data=llm_mrr_series,
        chart_type="line",
        height="stretch",
        delta_color="blue",
    )

if st.session_state.dataframe is not None:
    st.subheader("Data Preview")
    st.dataframe(st.session_state.dataframe)
