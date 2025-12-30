import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Question: A
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


# in this part we want to find the unique genres for the one-hot encoding for the movies
all_genres = sorted(
    {genre for sublist in movies['genres'] for genre in sublist})

# here we create the column of each genre for the one-hot encoding for the movies
for genre in all_genres:
    movies[genre] = movies["genres"].apply(lambda x: 1 if genre in x else 0)

# here we drop the genres column since its useless
movies = movies.drop("genres", axis=1)

# print(user_profile.head())

# Question: B
X = user_profile[["(no genres listed)", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
                  "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]]

# we have to scale the because the silhouette score uses distances, and not scaling will not work very well
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

silhouette_scores = {}
models = {}

# in this for loop we are checking each k in the kmeans and we are trying to see which one is the best
# by first finding each silhouette score for each k and then storing both the model and the score in a table/ list
for k in [3, 4, 5]:
    kmeans = KMeans(
        n_clusters=k,
        random_state=575,
        n_init='auto'
    )

    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    silhouette_scores[k] = score
    models[k] = kmeans

    # print(f"K = {k}, Silhouette Score = {score:.4f}")

# find the best k for the clustering
best_k = max(silhouette_scores, key=silhouette_scores.get)
best_model = models[best_k]

# print(f"Best k = {best_k}")

# assining the clusters to the users
user_profile['cluster'] = best_model.labels_

# we had the scaled the clusters and since they are scaled we have to "unscale" them
cluster_centers_scaled = best_model.cluster_centers_
cluster_centers = scaler.inverse_transform(cluster_centers_scaled)

cluster_profiles = pd.DataFrame(
    cluster_centers,
    # here we take every other column (aka genres) except the cluster one.
    columns=user_profile.columns.drop('cluster')
)

cluster_profiles.index.name = "cluster"
# print(cluster_profiles)

# naming the clusters as "xyz genre Fans"
cluster_names = {}

for cluster_id, row in cluster_profiles.iterrows():
    top_genre = row.idxmax()
    cluster_names[cluster_id] = f"{top_genre} Fans"

# putting the correct cluster names in the user profiles
# each user id has a cluster next to it
user_profile['cluster_name'] = user_profile['cluster'].map(cluster_names)

# Question: C
