# Taylor Swift Song Recommender

A simple ML-powered web app that recommends Taylor Swift songs based on audio features.

Built with **NumPy, pandas, and cosine similarity**, and deployed using **Streamlit**.

---

## Demo

[ts-recommender.streamlit.app](https://ts-recommender.streamlit.app/)

---

## Features

- Select any Taylor Swift song
- Get 5 similar song recommendations
- Uses real Spotify audio features:
  - danceability
  - energy
  - valence
  - acousticness
  - tempo
- Clean UI with Streamlit
- Duplicate songs automatically removed

---

## How It Works

1. **Data Processing**
   - Load Spotify dataset using pandas
   - Remove duplicate songs
   - Select key audio features

2. **Feature Scaling (NumPy)**
   ```python
   mean = np.mean(X, axis=0)
   std = np.std(X, axis=0)
   X_scaled = (X - mean) / std

3. **Vector Normalization**
    ```python
    norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
    X_normalized = X_scaled / norms 
4. **Cosine Similarity**
    ```python
    similarity_matrix = X_normalized @ X_normalized.T
5. **Recommendation**
    - Find the selected song
    - Rank all songs by similarity
    - Return top 5 unique recommendations

## Tech Stack
- Python
- NumPy
- pandas
- scikit-learn (optional baseline)
- Streamlit

## Running Locally
    ```python
    # clone repo
    git clone https://github.com/your-username/taylor-swift-recommender.git
    cd taylor-swift-recommender

    # create virtual environment
    python3 -m venv venv
    source venv/bin/activate

    # install dependencies
    pip install -r requirements.txt

    # run app
    streamlit run app.py

