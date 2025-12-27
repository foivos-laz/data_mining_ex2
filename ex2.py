import pandas as pd

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

unique_genres = (
    movies["genres"]
    .dropna()
    .str.split("|")
    .explode()
    .str.strip()
    .unique()
)

nr_unique_genres = unique_genres.size

users = []

print(ratings["userId"].max())

# for i in range(1, ratings["userId"].max(), 1):
