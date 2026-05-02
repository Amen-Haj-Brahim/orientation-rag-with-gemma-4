import streamlit as st

from main import configure_console, initialize_rag, load_config
from src.rag_pipeline import answer_question
from src.text_splitter import normalize_arabic_text


st.set_page_config(
    page_title="Orientation RAG",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_rag_pipeline(force_reindex: bool = False):
    configure_console()
    config = load_config()
    rag_pipeline, _ = initialize_rag(config, force_reindex=force_reindex)
    return rag_pipeline


def get_source_rows(source_documents):
    rows = []
    for doc in source_documents:
        metadata = doc.metadata
        rows.append(
            {
                "source": metadata.get("source_file", ""),
                "row": metadata.get("row_id", metadata.get("row_index", "")),
                "code": metadata.get("program_code", ""),
                "program": metadata.get("program_ar") or metadata.get("program_fr", ""),
                "university": metadata.get("university", ""),
                "establishment": metadata.get("faculty", ""),
                "type": metadata.get("doc_type", ""),
            }
        )
    return rows


st.title("Orientation RAG")
st.caption("Ask questions about programs, universities, scores, duration, and career paths.")

force_reindex = False
with st.sidebar:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me in Arabic, French, or English about the orientation guide.",
            "sources": [],
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                st.dataframe(message["sources"], width="stretch", hide_index=True)

question = st.chat_input("Ask a question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                rag_pipeline = get_rag_pipeline(force_reindex=force_reindex)
                result = answer_question(rag_pipeline, normalize_arabic_text(question))
                source_rows = get_source_rows(result.get("source_documents", []))
                st.write(result["answer"])
                timings = result.get("timings", {})
                if timings:
                    st.caption(
                        " | ".join(
                            f"{name.replace('_seconds', '')}: {value:.2f}s"
                            for name, value in timings.items()
                        )
                    )
                if source_rows:
                    with st.expander("Sources"):
                        st.dataframe(source_rows, width="stretch", hide_index=True)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": source_rows,
                    }
                )
            except Exception as exc:
                error = f"Error: {exc}"
                st.error(error)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error, "sources": []}
                )
