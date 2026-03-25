import streamlit as st
from recommender import load_data, recommend

df, similarity_matrix = load_data()
    
# UI 
st.title("Taylor Swift Song Recommender")
st.write("Pick a Taylor Swift song and get similar recommendations.")

song_list = sorted(df["name"].unique())
selected_song = st.selectbox("Choose a song:", song_list)

if st.button("Recommend songs"):
    recommendations = recommend(selected_song, df, similarity_matrix)

    if not recommendations:
        st.error("Song not found.")
    else:
        st.subheader(f"Recommendations for {selected_song}")
        for i, (song, album, score) in enumerate(recommendations, start=1):
            st.write(f"{i}. **{song}** — {album} ({score:.2f})")

