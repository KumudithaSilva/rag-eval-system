import streamlit as st

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

if "document" not in st.session_state:
    st.session_state.document = None


# --- SIDEBAR: Filters + Info ---
with st.sidebar:
    st.title(":material/filter_alt: Filters")

    # Collection Name
    collection_name = st.text_input("Collection Name")

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

    # Submit
    if st.button("Submit"):
        st.session_state.document = {
            "collection_name": collection_name,
            "chunking": {"type": chunking_type, "config": config},
            "embedding_model": embedding_model,
            "rag_preprocess": rag_preporcess,
        }

if st.session_state.document:
    st.subheader("Generated Config")
    st.json(st.session_state.document)
