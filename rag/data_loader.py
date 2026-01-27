import pandas as pd

def load_and_prepare_data(data_dir="data"):
    movies = pd.read_csv(f"{data_dir}/movies.csv")
    ratings = pd.read_csv(f"{data_dir}/ratings.csv")

    df = pd.merge(movies, ratings, on="movieId")
    df = df.drop(columns=["timestamp"])

    avg = (
        df.groupby("movieId", as_index=False)
        .agg(average_rating=("rating", "mean"))
    )
    avg["average_rating"] = avg["average_rating"].round(2)

    final = pd.merge(
        movies[["movieId", "title", "genres"]],
        avg,
        on="movieId",
        how="inner",
    )
    return final
