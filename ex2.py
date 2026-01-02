import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

TEST_SPLIT = 0.3
SEED = 123

# Question: A
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# here we are creating a custom splitting algorythm since the default one offered by the libraries above won't work
# given the instructions "Για την αποτίμηση των συστημάτων θα χρησιμοποιήσετε διαχωρισμό σε σύνολο εκπαίδευσης/ελέγχου ανά χρήστη"


def train_test_split_per_user(ratings, test_size=0.3, seed=42):
    np.random.seed(seed)
    train_parts = []
    test_parts = []

    # here we iterate using userId to comply with the requirement of the instuction (seen in the above comment before the def ...)
    for user_id, group in ratings.groupby("userId"):
        # here we mix the ratings of every user so we don't take the same first or last ones everytime we run the algorythm
        # basically making the selection of the ratings random, removing the bias of their appearence within the data set
        group = group.sample(frac=1, random_state=seed)
        # this computes the amount of test ratings by mutliplying their amount with the %test and then rounding it to the closest intS
        n_test = int(len(group) * test_size)

        # here we are seperating the data into test and train
        test_parts.append(group.iloc[:n_test])
        train_parts.append(group.iloc[n_test:])

    # for both test and train this part merges all of the above user based spliting in one and reorganises it
    # so it starts correctly
    return (
        pd.concat(train_parts).reset_index(drop=True),
        pd.concat(test_parts).reset_index(drop=True)
    )


# we are splitting the data here since if we do it later, we might have data leakage, as well as
# not evaluating the recommender systems very well or accurately
ratings_train, ratings_test = train_test_split_per_user(
    ratings, TEST_SPLIT, seed=SEED)

# here we want to split the genres of each movie and put it in the ratings table
# as to make it easier, because the current way they are formated isn't too helpful
movies["genres"] = movies["genres"].str.split("|")
movies_exploded = movies.explode("genres")
ratings_with_merged_genres = ratings_train.merge(movies_exploded, on="movieId")

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
