import json
import time
import pandas as pd
import requests
import streamlit as st
from streamlit_echarts import JsCode, st_echarts

API_URL = "http://127.0.0.1:8000/rag/mongo"
FILE_UPLOAD = "http://127.0.0.1:8000/rag/user_upload"

# --- Page Configuration ---
st.set_page_config(
    page_title="Streamlit RAG Evaluator",
    page_icon=":material/monitoring:",
    layout="wide",
)

st.title(":material/monitoring: RAG Evaluation")
st.markdown(
    "This dashboard presents a collection of RAG evaluation test data integrated into Streamlit apps"
    ", designed to help benchmark RAG techniques and uncover the most accurate and actionable insights.\n"
)

# --- Value Holder ---
config = {}
placeholder = st.empty()

if "dataframe" not in st.session_state:
    st.session_state.dataframe = None


# --- SIDEBAR: Filters + Info ---
with st.sidebar:
    st.title(":material/filter_alt: Filters")

    uploaded_folder = st.file_uploader(
        "Upload Knowledge Base",
        type=["zip", "rar"],
        accept_multiple_files=False,
        max_upload_size=50,
    )

    testtest_folder = st.file_uploader(
        "Upload Knowledge Base Testset",
        type=["zip", "rar"],
        accept_multiple_files=False,
        max_upload_size=50,
    )

    # Collection Name
    collection_name = st.text_input("Collection Name", value="rag_eval_2")

    # Chunking Type
    chunking_type = st.selectbox("Chunking Type", ["Default Chunking", "LLM Chunking"])

    # Conditional UI for chunking
    if chunking_type == "Default Chunking":
        config["chunk_size"] = st.number_input(
            "Chunk Size",
            min_value=100,
            value=1000,
            help="The maximum number of characters per chunk. "
            "Larger values mean fewer but bigger chunks.",
        )
        config["chunk_overlap"] = st.number_input(
            "Chunk Overlap",
            min_value=10,
            value=500,
            help="The number of characters shared between consecutive chunks. "
            "Helps preserve context across chunks.",
        )

    elif chunking_type == "LLM Chunking":
        config["llm_model"] = st.selectbox(
            "LLM Model",
            [
                "llama3.2",
                "openai/gpt-4o",
                "openai/gpt-4o-mini",
                "openai/gpt-4.1-nano",
                "google/gemini-2.5-pro",
            ],
        )

    # Embedding Provider
    embedding_provider = st.selectbox("Embedding Provider", ["HuggingFace", "OpenAI"])

    # Conditional UI for embedding provider
    if embedding_provider == "HuggingFace":
        config["emb_model_name"] = st.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "all-distilroberta-v1"],
        )

    elif embedding_provider == "OpenAI":
        config["emb_model_name"] = st.selectbox(
            "Embedding Model",
            ["text-embedding-3-small", "text-embedding-3-large"],
        )

    # Mongo Collection
    if collection_name and st.session_state.dataframe is None:
        try:
            response = requests.get(API_URL)
            response.raise_for_status()

            data = response.json().get("response", [])

            st.session_state.dataframe = pd.DataFrame(data)
        except Exception as e:
            st.error(f"Failed to load Mongo data: {e}")

    # Submit
    st.space("small")
    if st.button("Submit"):
        if not collection_name.strip():
            st.error("Collection Name is required")
        if not uploaded_folder or not testtest_folder:
            st.error("Uploading File is required")
        else:
            st.session_state.document = {
                "collection_name": collection_name,
                "chunking": {"type": chunking_type, "config": config},
                "embedding": {"type": embedding_provider, "config": config},
            }
            files = [
                ("file", (uploaded_folder.name, uploaded_folder, uploaded_folder.type)),
                ("file", (testtest_folder.name, testtest_folder, testtest_folder.type)),
            ]

            data = {"document": json.dumps(st.session_state.document)}

            response = requests.post(FILE_UPLOAD, files=files, data=data)

            st.space("small")

            placeholder.info(response.json().get("response", ""))
            time.sleep(5)
            placeholder.empty()


if st.session_state.dataframe is not None and st.session_state.dataframe.shape[0] >= 2:

    df = st.session_state.dataframe
    col1, col2, col3, col4 = st.columns(4)

    # --- BASELINE: Retrieval Performance Metrics ---

    with col1:
        # This metric clearly represents the latest MRR and how it compares to the historical max
        mrr = df["mrr"]

        latest_chunking_type = df["chunking.type"].iloc[-1]
        latest_mrr = df["mrr"].iloc[-1]
        top_mrr = df["mrr"].max()

        delta = latest_mrr - top_mrr

        st.metric(
            label=f"{latest_chunking_type} MRR (Latest)",
            value=f"{latest_mrr:.4f}",
            delta=f"{delta:+.4f}",
            border=True,
            chart_data=mrr,
            chart_type="line",
        )

    with col2:
        # This metric clearly represents the latest NDCG and how it compares to the historical max
        ndcg = df["ndcg"]

        latest_chunking_type = df["chunking.type"].iloc[-1]
        latest_ndcg = df["ndcg"].iloc[-1]
        top_ndcg = ndcg.max()

        delta = latest_mrr - top_mrr

        st.metric(
            label=f"{latest_chunking_type} NDCG (Latest)",
            value=f"{latest_ndcg:.4f}",
            delta=f"{delta:+.4f}",
            border=True,
            chart_data=ndcg,
            chart_type="line",
        )

    with col3:
        # This metric clearly represents the latest Recall and how it compares to the historical max
        recall = df["recall@K"]

        latest_chunking_type = df["chunking.type"].iloc[-1]
        latest_recall = df["recall@K"].iloc[-1]
        top_recall = recall.max()

        delta = latest_mrr - top_mrr

        st.metric(
            label=f"{latest_chunking_type} Recall (Latest)",
            value=f"{latest_recall:.4f}",
            delta=f"{delta:+.4f}",
            border=True,
            chart_data=recall,
            chart_type="line",
        )

    with col4:

        # This metric clearly represents the latest hitK and how it compares to the historical max
        hitK = df["hit@K"]

        latest_chunking_type = df["chunking.type"].iloc[-1]
        latest_hitK = df["hit@K"].iloc[-1]
        top_hitK = hitK.max()

        delta = latest_mrr - top_mrr

        st.metric(
            label=f"{latest_chunking_type} HitK (Latest)",
            value=f"{latest_hitK:.4f}",
            delta=f"{delta:+.4f}",
            border=True,
            chart_data=hitK,
            chart_type="line",
        )

    # --- MODEL EVAL: Default and LLM Based MRR ---
    st.subheader(":material/trending_up: Default and LLM Based Chunking")
    st.caption(
        "Track MRR and NDCG progress with Default chunking and LLM Based chunking."
    )

    col1, col2 = st.columns(2)

    with col1:

        filtered_llm_df = df[df["chunking.type"] == "LLM Chunking"]
        llm_mrr_series = filtered_llm_df["mrr"]

        filtered_default_df = df[df["chunking.type"] == "Default Chunking"]
        default_mrr_series = filtered_default_df["mrr"]

        # Prepare data for line chart
        steps = df.index.tolist()
        eval_llm_mrr = llm_mrr_series.tolist()
        eval_default_mrr = default_mrr_series.tolist()

        option = {
            "title": {
                "text": "MRR Comparison",
                "left": "center",
                "top": 15,
                "textStyle": {"fontSize": 22, "fontWeight": "bold"},
            },
            "grid": {
                "left": "5%",  # reduce left margin to push chart left
                "right": "15%",  # increase right margin
                "top": "15%",
                "bottom": "20%",
            },
            "toolbox": {
                "feature": {
                    "saveAsImage": {},
                    "dataView": {"readOnly": True},
                    "restore": {},
                    "magicType": {"type": ["line", "bar"]},  # only one toolbox key
                }
            },
            "tooltip": {
                "trigger": "axis",
                "formatter": JsCode(
                    """
                function(params){
                    // params is an array of series info at this axis point
                    var result = params[0].axisValue + '<br/>';
                    params.forEach(function(item){
                        result += item.marker + item.seriesName + ': ' + item.data + '<br/>';
                    });
                    return result;
                }
                """
                ),
            },
            "legend": {"data": ["LLM Based", "Default"], "bottom": 10},
            "xAxis": {"type": "category", "data": steps, "name": "Step"},
            "yAxis": {"type": "value", "name": "MRR"},
            "series": [
                {
                    "name": "LLM Based",
                    "type": "line",
                    "data": eval_llm_mrr,
                    "smooth": True,
                },
                {
                    "name": "Default",
                    "type": "line",
                    "smooth": True,
                    "areaStyle": {"opacity": 0.1},
                    "data": eval_default_mrr,
                },
            ],
        }
        st_echarts(
            options=option,
            height="400px",
            width="700px",
            key="trend_mrr",
            theme="streamlit",
        )

    with col2:

        filtered_llm_df = df[df["chunking.type"] == "LLM Chunking"]
        llm_mrr_series = filtered_llm_df["ndcg"]

        filtered_default_df = df[df["chunking.type"] == "Default Chunking"]
        default_mrr_series = filtered_default_df["ndcg"]

        # Prepare data for line chart
        steps = df.index.tolist()
        eval_llm_ndcg = llm_mrr_series.tolist()
        eval_default_ndcg = default_mrr_series.tolist()

        option = {
            "title": {
                "text": "NDCG Comparison",
                "left": "center",
                "top": 15,
                "textStyle": {"fontSize": 22, "fontWeight": "bold"},
            },
            "grid": {
                "left": "5%",  # reduce left margin to push chart left
                "right": "15%",  # increase right margin
                "top": "15%",
                "bottom": "20%",
            },
            "toolbox": {
                "feature": {
                    "saveAsImage": {},
                    "dataView": {"readOnly": True},
                    "restore": {},
                    "magicType": {"type": ["line", "bar"]},  # only one toolbox key
                }
            },
            "tooltip": {
                "trigger": "axis",
                "formatter": JsCode(
                    """
                function(params){
                    // params is an array of series info at this axis point
                    var result = params[0].axisValue + '<br/>';
                    params.forEach(function(item){
                        result += item.marker + item.seriesName + ': ' + item.data + '<br/>';
                    });
                    return result;
                }
                """
                ),
            },
            "legend": {"data": ["LLM Based", "Default"], "bottom": 10},
            "xAxis": {"type": "category", "data": steps, "name": "Step"},
            "yAxis": {"type": "value", "name": "MRR"},
            "series": [
                {
                    "name": "LLM Based",
                    "type": "line",
                    "data": eval_llm_ndcg,
                    "smooth": True,
                },
                {
                    "name": "Default",
                    "type": "line",
                    "smooth": True,
                    "areaStyle": {"opacity": 0.1},
                    "data": eval_default_ndcg,
                },
            ],
        }
        st_echarts(
            options=option,
            height="400px",
            width="700px",
            key="trend_ndgc",
            theme="streamlit",
        )

    # --- MODEL EVAL: Embedding Models and Chunking Type ---
    st.subheader(":material/link: Embedding Model & Chunk Type")
    st.caption("Track MRR and NDCG progress with embedding models and chunking types.")

    col1, col2 = st.columns(2)

    with col1:
        df = st.session_state.dataframe

        pivot_df = df.pivot_table(
            index="embedding.type",
            columns="chunking.type",
            values="mrr",
            aggfunc="median",
        )

        x_labels = list(pivot_df.columns)  # chunking types (x-axis)
        y_labels = list(pivot_df.index)  # embedding models (y-axis)

        data = []
        for i, emb in enumerate(y_labels):  # y-axis
            for j, chunk in enumerate(x_labels):  # x-axis
                val = pivot_df.loc[emb, chunk]
                data.append([j, i, float(val) if pd.notna(val) else None])

        option = {
            "title": {"text": "MRR", "left": "center"},
            "tooltip": {"position": "top", "formatter": "{c}"},
            "toolbox": {
                "feature": {
                    "saveAsImage": {},
                    "dataView": {"readOnly": True},
                    "restore": {},
                }
            },
            "grid": {
                "left": "5%",
                "right": "15%",
                "top": "15%",
                "bottom": "25%",
            },
            "xAxis": {
                "type": "category",
                "data": x_labels,
                "name": "Chunking Type",
                "axisLabel": {"rotate": 30, "color": "#ffffff"},
                "nameTextStyle": {"color": "#ffffff"},
            },
            "yAxis": {
                "type": "category",
                "data": y_labels,
                "name": "Embedding Model",
                "axisLabel": {"color": "#ffffff"},
                "nameTextStyle": {"color": "#ffffff"},
            },
            "visualMap": {
                "min": 0,
                "max": 1,
                "calculable": True,
                "orient": "horizontal",
                "left": "center",
                "textStyle": {"color": "#ffffff"},
                "inRange": {"color": ["#F7F8F0", "#9CD5FF", "#7AAACE", "#355872"]},
                "precision": 2,
            },
            "series": [
                {
                    "name": "MRR",
                    "type": "heatmap",
                    "data": data,
                    "label": {"show": True},
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowColor": "rgba(0,0,0,0.5)",
                        }
                    },
                }
            ],
        }

        st_echarts(options=option, height="500px", width="700px", theme="streamlit")

    with col2:
        df = st.session_state.dataframe

        pivot_df = df.pivot_table(
            index="embedding.type",
            columns="chunking.type",
            values="ndcg",
            aggfunc="median",
        )

        x_labels = list(pivot_df.columns)  # chunking types (x-axis)
        y_labels = list(pivot_df.index)  # embedding models (y-axis)

        data = []
        for i, emb in enumerate(y_labels):  # y-axis
            for j, chunk in enumerate(x_labels):  # x-axis
                val = pivot_df.loc[emb, chunk]
                data.append([j, i, float(val) if pd.notna(val) else None])

        option = {
            "title": {"text": "NDCG", "left": "center"},
            "tooltip": {"position": "top", "formatter": "{c}"},
            "toolbox": {
                "feature": {
                    "saveAsImage": {},
                    "dataView": {"readOnly": True},
                    "restore": {},
                }
            },
            "grid": {
                "left": "5%",
                "right": "15%",
                "top": "15%",
                "bottom": "25%",
            },
            "xAxis": {
                "type": "category",
                "data": x_labels,
                "name": "Chunking Type",
                "axisLabel": {"rotate": 30, "color": "#ffffff"},
                "nameTextStyle": {"color": "#ffffff"},
            },
            "yAxis": {
                "type": "category",
                "data": y_labels,
                "name": "Embedding Model",
                "axisLabel": {"color": "#ffffff"},
                "nameTextStyle": {"color": "#ffffff"},
            },
            "visualMap": {
                "min": 0,
                "max": 1,
                "calculable": True,
                "orient": "horizontal",
                "left": "center",
                "textStyle": {"color": "#ffffff"},
                "inRange": {"color": ["#F7F8F0", "#9CD5FF", "#7AAACE", "#355872"]},
                "precision": 2,
            },
            "series": [
                {
                    "name": "MRR",
                    "type": "heatmap",
                    "data": data,
                    "label": {"show": True},
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowColor": "rgba(0,0,0,0.5)",
                        }
                    },
                }
            ],
        }

        st_echarts(options=option, height="500px", width="700px", theme="streamlit")

    # --- Data Preview ---
    st.space("medium")
    with st.expander("Raw data preview", icon=":material/table_view:"):
        st.dataframe(st.session_state.dataframe.head(5), width="stretch")
else:
    st.markdown(
        """
    <div style="
        border: 1px solid #ffffff; 
        padding: 50px; 
        border-radius: 5px; 
        font-weight: bold;
        word-wrap: break-word;
        text-align: center;       
        display: flex;
        align-items: center;      
        justify-content: center;  
        height: 150px;            
    ">
    The DataFrame has fewer rows. Metrics cannot be displayed.
    </div>
    """,
        unsafe_allow_html=True,
    )
