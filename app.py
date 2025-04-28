import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('final_df.csv')
    return df

# Load the embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for plot descriptions
@st.cache_data
def create_embeddings(texts, _model):
    # Handle NaN values by replacing them with empty strings
    texts = [str(text) if not pd.isna(text) else "" for text in texts]
    return _model.encode(texts)

def main():
    st.title("Movie Recommendation System")
    
    # Load data and model
    df = load_data()
    model = load_model()
    
    # Create sidebar for filters
    st.sidebar.header("Filter Options")
    
    # Year filter as expandable section
    with st.sidebar.expander("Year Filter", expanded=False):
        min_year = int(df['Year'].min())
        max_year = int(df['Year'].max())
        year_range = st.slider(
            "Select Year Range",
            min_year, max_year,
            (min_year, max_year)
        )
    
    # Genre filter as expandable section
    with st.sidebar.expander("Genre Filter", expanded=False):
        all_genres = set()
        for genres in df['Genre'].str.split(', '):
            if isinstance(genres, list):
                all_genres.update(genres)
        selected_genre = st.multiselect(
            "Select Genre(s)",
            sorted(list(all_genres))
        )
    
    # Runtime filter as expandable section
    with st.sidebar.expander("Runtime Filter", expanded=False):
        min_runtime = int(df['runtimeMinutes'].min())
        max_runtime = int(df['runtimeMinutes'].max())
        runtime_range = st.slider(
            "Select Runtime Range (minutes)",
            min_runtime, max_runtime,
            (min_runtime, max_runtime)
        )
    
    # Directors filter as expandable section
    with st.sidebar.expander("Director Filter", expanded=False):
        directors = sorted(df['Director'].dropna().unique())
        selected_directors = st.multiselect("Select Director(s)", directors)
    
    # Actors filter as expandable section
    with st.sidebar.expander("Actor Filter", expanded=False):
        # Extract all actors and create a unique list
        all_actors = set()
        for actors in df['Actors'].dropna().str.split(', '):
            if isinstance(actors, list):
                all_actors.update(actors)
        selected_actors = st.multiselect("Select Actor(s)", sorted(list(all_actors)))
    
    # Language filter as expandable section
    with st.sidebar.expander("Language Filter", expanded=False):
        languages = sorted(df['Language'].str.split(', ').explode().dropna().unique())
        selected_language = st.selectbox("Select Language", [""] + languages)
    
    # Country filter as expandable section
    with st.sidebar.expander("Country Filter", expanded=False):
        countries = sorted(df['Country'].str.split(', ').explode().dropna().unique())
        selected_country = st.selectbox("Select Country", [""] + countries)
    
    # Ratings filters as expandable section
    with st.sidebar.expander("Rating Filters", expanded=False):
        # Average rating minimum
        min_rating = st.slider(
            "Minimum Average Rating",
            0.0, 10.0, 0.0, 0.1
        )
        
        # IMDB rating filter
        min_imdb = st.slider(
            "Minimum IMDB Rating",
            0.0, 10.0, 0.0, 0.1
        )
        
        # Rotten Tomatoes rating filter
        min_rt = st.slider(
            "Minimum Rotten Tomatoes Rating",
            0.0, 10.0, 0.0, 0.1
        )
        
        # Metascore filter
        min_meta = st.slider(
            "Minimum Metascore",
            0.0, 10.0, 0.0, 0.1
        )
    
    # Content rating filters as expandable section
    with st.sidebar.expander("Content Rating Filters", expanded=False):
        # Content rating codes
        rating_codes = [
            ('sex_code', 'Sexual Content'), 
            ('violence_code', 'Violence'), 
            ('profanity_code', 'Profanity'), 
            ('drug_code', 'Drug Use'), 
            ('intense_code', 'Intense Scenes')
        ]
        
        code_filters = {}
        
        for code, label in rating_codes:
            min_code = float(df[code].min())
            max_code = float(df[code].max())
            code_range = st.slider(
                f"{label} Range (1-5)",
                min_code, max_code,
                (min_code, max_code)
            )
            code_filters[code] = code_range
    
    # Main content area
    st.header("Describe the movie you're looking for")
    search_text = st.text_area("Enter description (optional)", "")
    
    # Sort options in main content area (not sidebar)
    col1, col2 = st.columns([3, 1])
    with col1:
        sort_by = st.selectbox(
            "Sort Results By",
            ["Relevance (if search used)", "Average Rating", "IMDB Rating", "Release Year (newest)", "Runtime"]
        )
    
    # Add limit results in the main content as a dropdown
    with col2:
        results_limit = st.selectbox("Results Limit", [5, 10, 20, 30, 50, 100])
    
    # Search button
    search_clicked = st.button("Search Movies", type="primary")
    
    # Only process when search button is clicked
    if search_clicked:
        # Apply filters
        filtered_df = df.copy()
        
        # Year filter
        if year_range != (min_year, max_year):
            filtered_df = filtered_df[filtered_df['Year'].between(year_range[0], year_range[1])]
        
        # Runtime filter
        if runtime_range != (min_runtime, max_runtime):
            filtered_df = filtered_df[filtered_df['runtimeMinutes'].between(runtime_range[0], runtime_range[1])]
        
        # Rating filters
        if min_rating > 0:
            filtered_df = filtered_df[filtered_df['averageRating'] >= min_rating]
        
        if min_imdb > 0:
            filtered_df = filtered_df[filtered_df['imdbRating'] >= min_imdb]
        
        if min_rt > 0:
            filtered_df = filtered_df[filtered_df['RottenTomatoes_10'] >= min_rt]
            
        if min_meta > 0:
            filtered_df = filtered_df[filtered_df['Metascore_10'] >= min_meta]
        
        # Genre filter
        if selected_genre:
            filtered_df = filtered_df[filtered_df['Genre'].apply(
                lambda x: any(genre in str(x).split(', ') for genre in selected_genre)
            )]
        
        # Director filter
        if selected_directors:
            filtered_df = filtered_df[filtered_df['Director'].apply(
                lambda x: any(director in str(x) for director in selected_directors)
            )]
        
        # Actor filter
        if selected_actors:
            filtered_df = filtered_df[filtered_df['Actors'].apply(
                lambda x: any(actor in str(x) for actor in selected_actors)
            )]
        
        # Language filter
        if selected_language:
            filtered_df = filtered_df[filtered_df['Language'].str.contains(selected_language, case=False, na=False)]
        
        # Country filter
        if selected_country:
            filtered_df = filtered_df[filtered_df['Country'].str.contains(selected_country, case=False, na=False)]
        
        # Content rating filters - apply only if they've been changed from default
        for code, label in rating_codes:
            min_val, max_val = code_filters[code]
            min_code = float(df[code].min())
            max_code = float(df[code].max())
            
            if (min_val, max_val) != (min_code, max_code):
                filtered_df = filtered_df[
                    filtered_df[code].between(min_val, max_val)
                ]
        
        # Semantic search if text is provided
        if search_text.strip():
            # Make sure there are results before attempting search
            if len(filtered_df) > 0:
                # Create embeddings for filtered plots and search query
                plot_embeddings = create_embeddings(filtered_df['Plot'].tolist(), model)
                query_embedding = model.encode([search_text])
                
                # Calculate similarities
                similarities = cosine_similarity(query_embedding, plot_embeddings)[0]
                
                # Add similarities to dataframe
                filtered_df = filtered_df.copy()
                filtered_df['similarity'] = similarities
                
                # Sort by similarity
                filtered_df = filtered_df.sort_values('similarity', ascending=False)
            else:
                st.warning("No movies match your filters. Try broadening your search criteria.")
        
        # Apply sorting
        if sort_by == "Average Rating" and not (search_text.strip() and sort_by == "Relevance (if search used)"):
            filtered_df = filtered_df.sort_values('averageRating', ascending=False)
        elif sort_by == "IMDB Rating" and not (search_text.strip() and sort_by == "Relevance (if search used)"):
            filtered_df = filtered_df.sort_values('imdbRating', ascending=False)
        elif sort_by == "Release Year (newest)" and not (search_text.strip() and sort_by == "Relevance (if search used)"):
            filtered_df = filtered_df.sort_values('Year', ascending=False)
        elif sort_by == "Runtime" and not (search_text.strip() and sort_by == "Relevance (if search used)"):
            filtered_df = filtered_df.sort_values('runtimeMinutes', ascending=False)
        # If relevance and search used, already sorted by similarity
        
        # Limit results
        filtered_df = filtered_df.head(results_limit)
        
        # Display results
        st.header(f"Found {len(filtered_df)} movies")
        
        for _, movie in filtered_df.iterrows():
            st.subheader(f"{movie['Title']} ({int(movie['Year'])})")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Plot:** {movie['Plot']}")
                st.write(f"**Director:** {movie['Director']}")
                st.write(f"**Actors:** {movie['Actors']}")
                st.write(f"**Genre:** {movie['Genre']}")
                st.write(f"**Language:** {movie['Language']}")
                st.write(f"**Country:** {movie['Country']}")
            
            with col2:
                st.write(f"**Average Rating:** {movie['averageRating']:.1f}/10")
                st.write(f"**IMDB:** {movie['imdbRating']}/10")
                if not pd.isna(movie['RottenTomatoes_10']):
                    st.write(f"**Rotten Tomatoes:** {movie['RottenTomatoes_10']}/10")
                if not pd.isna(movie['Metascore_10']):
                    st.write(f"**Metascore:** {movie['Metascore_10']}/10")
                st.write(f"**Runtime:** {int(movie['runtimeMinutes'])} minutes")
                if search_text.strip() and 'similarity' in filtered_df.columns:
                    st.write(f"**Relevance:** {movie['similarity']:.2f}")
                
                # Add IMDB link
                imdb_id = movie['imdbID']
                if not pd.isna(imdb_id):
                    imdb_url = f"https://www.imdb.com/title/{imdb_id}/"
                    st.markdown(f"[View on IMDB]({imdb_url})")
            
            st.divider()

if __name__ == "__main__":
    main()
