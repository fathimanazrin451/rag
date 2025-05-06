# app.py
# Streamlit app for RAG-based Title & Abstract Screening
# Requirements:
!pip install -q streamlit faiss-cpu sentence-transformers transformers scikit-learn openpyxl
# Usage:
#streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Load models (cached)
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
    model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return embedding_model, tokenizer, model, device

embedding_model, tokenizer_rag, model_rag, device = load_models()

# 2. App title
st.set_page_config(page_title="RAG Screening App", layout="wide")
st.title("📰 RAG-Based Title & Abstract Screening")

# 3. Sidebar: file upload and query
st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload an Excel file", type=["xlsx"])

st.sidebar.header("2. Define Screening Query & Settings")
query = st.sidebar.text_input("Enter your screening query:")
k = st.sidebar.slider("Number of papers to retrieve", 1, 10, 5)

if uploaded_file and query:
    # 4. Load data
    df = pd.read_excel(uploaded_file)
    if 'Title' not in df.columns or 'Abstract' not in df.columns:
        st.error("Excel must contain 'Title' and 'Abstract' columns.")
    else:
        # 5. Prepare corpus & FAISS
        df = df.dropna(subset=['Title', 'Abstract']).reset_index(drop=True)
        corpus = (df['Title'] + " " + df['Abstract']).tolist()
        embeddings = embedding_model.encode(corpus, convert_to_numpy=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        # 6. Retrieval
        query_emb = embedding_model.encode(query, convert_to_numpy=True)
        _, indices = index.search(np.array([query_emb]), k)
        retrieved_idxs = indices[0]

        # 7. Generation
        context = " ".join([corpus[i] for i in retrieved_idxs])
        prompt = (
            f"Based on the following papers, decide whether to include or exclude them "
            f"for the query: '{query}'. Just return 'Include' or 'Exclude'. Context: {context}"
        )
        inputs = tokenizer_rag(prompt, return_tensors='pt', truncation=True, max_length=1024).to(device)
        outputs = model_rag.generate(**inputs, max_new_tokens=10)
        decision = tokenizer_rag.decode(outputs[0], skip_special_tokens=True).strip()

        # 8. Display results
        st.header("🔍 Screening Decision")
        st.subheader(decision)

        st.header(f"📄 Top {k} Retrieved Papers")
        results = []
        for idx in retrieved_idxs:
            title = df.loc[idx, 'Title']
            abstract = df.loc[idx, 'Abstract']
            st.markdown(f"**Title:** {title}")
            st.markdown(f"**Abstract:** {abstract}")
            st.markdown(f"**Prediction:** {decision}")
            st.markdown("---")
            results.append({
                'Title': title,
                'Abstract': abstract,
                'Prediction': decision
            })

        # 9. Download results
        result_df = pd.DataFrame(results)
        st.download_button(
            label="Download results as Excel",
            data=result_df.to_excel(index=False),
            file_name="rag_screening_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    if not uploaded_file:
        st.info("Please upload an Excel file to begin.")
    elif not query:
        st.info("Please enter a screening query.")
