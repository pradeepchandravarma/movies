import pandas as pd
from langchain_core.documents import Document

def build_documents(movies_csv, ratings_csv):
    df_movies = pd.read_csv(movies_csv)
    df_ratings = pd.read_csv(ratings_csv)

    df = (
        pd.merge(df_movies, df_ratings)
        .groupby("movieId", as_index=False)
        .agg(
            title=("title", "first"),
            genres=("genres", "first"),
            avg_rating=("rating", "mean"),
        )
    )

    return [
        Document(
            page_content=(
                f"{r.title}\n"
                f"Genres: {r.genres.replace('|', ', ')}\n"
                f"Average rating: {round(r.avg_rating, 2)}"
            ),
            metadata={"movieId": int(r.movieId)},
        )
        for r in df.itertuples()
    ]
