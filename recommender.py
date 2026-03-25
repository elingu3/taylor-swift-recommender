import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


scalar = StandardScaler()

def load_data():
    df = pd.read_csv("taylor_swift_spotify.csv")
    # keep only the columns you need
    df = df[["name", "album", "danceability", "energy", "valence", "acousticness", "tempo"]].copy()

    # normalize titles for duplicate removal
    df["clean_name"] = df["name"].str.lower().str.strip()

    # remove duplicate song titles, keeping the first one
    df = df.drop_duplicates(subset="clean_name").reset_index(drop=True)

    features = ["danceability", "energy", "valence", "acousticness", "tempo"]
    X = df[features].to_numpy()

    # scale manually because features have different values
    mean = np.mean(X, axis=0) # compute columnwise
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std # subtract mean from each column, divide by std

    # normalize vectors
    norms = np.linalg.norm(X_scaled, axis=1, keepdims=True) # compute length of each song vector
    X_normalized = X_scaled / norms # divide each row by its length

    # Compute song similarity
    similarity_matrix = X_normalized @ X_normalized.T # matrix multiplication, multipy by transpose
    # similarity_matrix[i][j] = similarity between song i and song j
    return df, similarity_matrix

# Recommendation function
def recommend(song_name, df, similarity_matrix, num_recommendations=5):
    # normalize input
    song_name = song_name.lower()

    # find song index

    song_column = df["name"] # get the column
    lowercase_songs = song_column.str.lower() # lowercase all song names
    matches_mask = lowercase_songs == song_name # compare each song to input
    matches = df[matches_mask] # filter the dataframe

    if matches.empty:
        print("Invalid song.")
        return
    
    idx = matches.index[0]

    # get similarity scores
    similarity_scores = similarity_matrix[idx]

    # sort highest to lowest
    scores = list(enumerate(similarity_scores))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # return top 5
    print(f"\nRecommendations for '{df.iloc[idx]['name']}':\n")

    recommendations = []
    recommended_titles = set()
    count = 0

    for song_idx, score in scores:
        clean_title = df.iloc[song_idx]["clean_name"]
        song_title = df.iloc[song_idx]["name"]
        album = df.iloc[song_idx]["album"]

        if clean_title == song_name:
            continue
        
        if clean_title in recommended_titles:
            continue
        
        recommended_titles.add(clean_title)
        recommendations.append((song_title, album, score))
        count += 1

        if len(recommendations) == num_recommendations:
            break
        
    return recommendations