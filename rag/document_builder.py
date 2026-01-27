from langchain_core.documents import Document

def build_documents(df):
    docs = []

    for _, row in df.iterrows():
        rating = float(row["average_rating"])

        if rating >= 4.0:
            bucket = "very_high"
        elif rating >= 3.5:
            bucket = "high"
        elif rating >= 3.0:
            bucket = "medium"
        else:
            bucket = "low"

        text = (
            f"{row['title']}\n"
            f"Genres: {row['genres'].replace('|', ', ')}\n"
            f"Average rating: {rating}\n"
            f"Rating bucket: {bucket}"
        )

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "movieId": int(row["movieId"]),
                    "title": row["title"],
                    "genres": row["genres"],
                    "average_rating": rating,
                    "rating_bucket": bucket,
                },
            )
        )
    return docs
