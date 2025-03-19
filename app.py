import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load Data
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    df = pd.read_csv('final_df.csv')
    embeddings = np.load('movie_embeddings.npy')
    df['embedding'] = embeddings.tolist()
    return df

# Semantic search function
def semantic_search(query, df, model, top_k=10):
    query_embedding = model.encode(query)
    df['similarity'] = df['embedding'].apply(lambda x: util.cos_sim(x, query_embedding).item())
    return df.sort_values(by='similarity', ascending=False).head(top_k)

# Filtering function based on provided criteria
def filter_movies(df, min_year, max_year, min_rating, sex_code, violence_code, profanity_code, drug_code, intense_code):
    df_filtered = df.copy()

    if min_year:
        df_filtered = df_filtered[df_filtered['Year'] >= min_year]
    if max_year:
        df_filtered = df_filtered[df_filtered['Year'] <= max_year]
    if min_rating:
        df_filtered = df_filtered[df_filtered['imdbRating'] >= min_rating]
    if sex_code is not None:
        df_filtered = df_filtered[df_filtered['sex_code'] <= sex_code]
    if violence_code is not None:
        df_filtered = df_filtered[df_filtered['violence_code'] <= violence_code]
    if profanity_code is not None:
        df_filtered = df_filtered[df_filtered['profanity_code'] <= profanity_code]
    if drug_code is not None:
        df_filtered = df_filtered[df_filtered['drug_code'] <= drug_code]
    if intense_code is not None:
        df_filtered = df_filtered[df_filtered['intense_code'] <= intense_code]

    df_filtered = df_filtered[(df_filtered['Year'] >= min_year) & (df_filtered['Year'] <= max_year)]

    return df_filtered

# Streamlit UI
st.title("🎬 Movie Recommendation Tool")

# Load data and model
df = load_data()
model = load_model()

# Sidebar Filters
st.sidebar.header("🎯 Movie Filters")
min_year, max_year = st.sidebar.slider('Year Range', int(df['Year'].min()), int(df['Year'].max()), (2000, 2024))
min_rating = st.sidebar.slider('Minimum IMDb Rating', 0.0, 10.0, 5.0)
sex_code = st.sidebar.slider("Max Sex/Nudity", 0, 4, 4)
violence_code = st.sidebar.slider("Max Violence", 0, 4, 4)
profanity_code = st.sidebar.slider("Max Profanity", 0, 4, 4)
drug_code = st.sidebar.slider("Max Drug Use", 0, 4, 4)
intense_code = st.sidebar.slider("Max Intense Scenes", 0, 4, 4)

query = st.text_input("Describe the kind of movie you want to watch:")

if st.button('Search'):
    df_filtered = filter_movies(df, min_year, max_year, min_rating, sex_code, violence_code, profanity_code, drug_code, intense_code)

    if query.strip():
        results = semantic_search(query, df_filtered, model)
    else:
        results = df_filtered.head(10)

    if not results.empty:
        for _, row in results.iterrows():
            st.subheader(f"{row['Title']} ({row['Year']})")
            st.write(f"**IMDb Rating:** {row['imdbRating']} | **Genre:** {row['Genre']}")
            st.write(f"**Plot:** {row['Plot']}")
            st.markdown(f"[IMDb Link](https://www.imdb.com/title/{row['imdbID']})")
            st.divider()
    else:
        st.warning("No movies found matching your criteria.")
