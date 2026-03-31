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
        "Upload Main Folder",
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
        config["k"] = st.number_input("K Value", min_value=1, value=3)
        config["size"] = st.number_input("Chunk Size", min_value=1, value=8)

    elif chunking_type == "LLM Chunking":
        config["k"] = st.number_input("K Value", min_value=1, value=3)
        config["llm_model"] = st.selectbox(
            "LLM Model",
            [
                "ollama/llama3.2",
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
        if not uploaded_folder:
            st.error("Uploading File is required")
        else:
            st.session_state.document = {
                "collection_name": collection_name,
                "chunking": {"type": chunking_type, "config": config},
                "embedding": {"type": embedding_provider, "config": config},
            }
            files = {
                "file": (uploaded_folder.name, uploaded_folder, uploaded_folder.type)
            }
            data = {"document": json.dumps(st.session_state.document)}

            response = requests.post(FILE_UPLOAD, files=files, data=data)

            st.space("small")

            placeholder.info(response.json().get("response", ""))
            time.sleep(5)
            placeholder.empty()


if st.session_state.dataframe is not None and st.session_state.dataframe.shape[0] >= 2:

    col1, col2, col3, col4 = st.columns(4)

    # --- BASELINE: Retrieval Performance Metrics ---
    with col1:

        mrr = st.session_state.dataframe["eval_default_mrr"]

        current = mrr.iloc[-1]
        top_value = mrr.max()

        delta = current - top_value

        st.metric(
            label="Default MRR (Max)",
            value=f"{current:.4f}",
            delta=f"{delta:+.4f}",
            border=True,
            chart_data=mrr,
            chart_type="line",
        )

    # Recent LLM performance changes
    with col2:

        llm_mrr_series = st.session_state.dataframe["eval_llm_mrr"]
        current = llm_mrr_series.iloc[-1]
        top_value = llm_mrr_series.max()

        delta = current - top_value

        st.metric(
            label="LLM MRR (Max)",
            value=f"{current:.2f}",
            delta=f"{delta:+.2f}",
            border=True,
            chart_data=llm_mrr_series,
            chart_type="area",
        )

    # Actual improvement from using LLM
    with col3:

        delta_mrr = (
            st.session_state.dataframe["eval_llm_mrr"]
            - st.session_state.dataframe["eval_default_mrr"]
        )

        current = delta_mrr.iloc[-1]
        previous = delta_mrr.iloc[-2]

        st.metric(
            label="Δ MRR (LLM vs Default)",
            value=f"{current:+.4f}",
            delta=f"{current - previous:+.4f}",
            border=True,
            chart_data=delta_mrr,
            chart_type="line",
        )

    with col4:

        delta_ndcg = (
            st.session_state.dataframe["eval_llm_ndcg"]
            - st.session_state.dataframe["eval_default_ndgc"]
        )

        current = delta_mrr.iloc[-1]
        previous = delta_mrr.iloc[-2]

        st.metric(
            label="Δ NDCG (LLM vs Default)",
            value=f"{current:+.4f}",
            delta=f"{current - previous:+.4f}",
            border=True,
            chart_data=delta_ndcg,
            chart_type="line",
        )

    # --- MODEL EVAL: Default and LLM Based MRR ---
    st.subheader(":material/trending_up: Default and LLM Based Chunking")
    st.caption(
        "Track MRR and NDCG progress with Default chunking and LLM Based chunking."
    )

    col1, col2 = st.columns(2)

    with col1:
        df = st.session_state.dataframe

        # Prepare data for line chart
        steps = df.index.tolist()
        eval_llm_mrr = df["eval_llm_mrr"].tolist()
        eval_default_mrr = df["eval_default_mrr"].tolist()

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
        df = st.session_state.dataframe

        # Prepare data for line chart
        steps = df.index.tolist()
        eval_llm_ndcg = df["eval_llm_ndcg"].tolist()
        eval_default_ndcg = df["eval_default_ndgc"].tolist()

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
            index="embedding_model",
            columns="chunking.type",
            values="eval_llm_mrr",
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
            index="embedding_model",
            columns="chunking.type",
            values="eval_llm_ndcg",
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
