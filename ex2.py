import pandas as pd

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# here we want to split the genres of each movie and put it in the ratings table
# as to make it easier, because the current way they are formated isn't too helpful
movies["genres"] = movies["genres"].str.split("|")
movies_exploded = movies.explode("genres")
ratings_with_merged_genres = ratings.merge(movies_exploded, on="movieId")

# here we create the users as data frames that have the userID and the genres with their means.
# it is important to have it as such, as to not lose the information of which genre is which with traditional list.
# the avoided lists would also make using the users' profile very very hard, and much more complex during both creation and use.
user_profile = ratings_with_merged_genres.groupby(
    ["userId", "genres"])["rating"].mean().unstack(fill_value=0)
