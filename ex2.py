import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

# Question: A
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# here we are creating a custom splitting algorythm since the default one offered by the libraries above won't work
# given the instructions "Για την αποτίμηση των συστημάτων θα χρησιμοποιήσετε διαχωρισμό σε σύνολο εκπαίδευσης/ελέγχου ανά χρήστη"


def train_test_split_per_user(ratings, test_size, seed):
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

# Question: C

# in this function to calculate the pearson similarity, which we will use later


def pearson_similarity(u1, u2):
    mask = (u1 != 0) & (u2 != 0)
    if np.sum(mask) < 2:
        return 0.0
    if np.std(u1[mask]) == 0 or np.std(u2[mask]) == 0:
        return 0.0
    return np.corrcoef(u1[mask], u2[mask])[0, 1]

# Σύστημα βάση περιεχομένου:


def recommend_cb(user_profile_row, movie_profiles, K):
    # we create the vectors needed for the content based recommender to work
    # and also we make the 1D user_profile into a 2D vector
    user_vec = user_profile_row[all_genres].values.reshape(1, -1)
    movie_vecs = movie_profiles[all_genres].values

    # in this part of the recommender we find the cos similarities, which we then turn the 2D array into a 1D one
    # then to find the top movies we sort them and
    # then we inverse them to be in descending order and we pic the K highest rated ones
    similarities = cosine_similarity(user_vec, movie_vecs).flatten()
    top_indices = similarities.argsort()[::-1][:K]

    return movie_profiles.iloc[top_indices].assign(score=similarities[top_indices])

# for the rest of the recommnder systems I will keep comments sparse for code legibility

# Σύστημα συνεργατικού φιλτραρίσματος με βάση τους N κοντινότερους χρήστες:


def recommend_cf(user_id, ratings_train, N, K):
    user_ratings = ratings_train[ratings_train['userId'] == user_id]
    user_mean = user_ratings['rating'].mean()

    other_users = ratings_train[ratings_train['userId']
                                != user_id]['userId'].unique()

    sims = []
    for other_id in other_users:
        other_ratings = ratings_train[ratings_train['userId'] == other_id]
        merged = pd.merge(user_ratings, other_ratings,
                          on='movieId', suffixes=('_u', '_v'))
        if len(merged) < 2:  # we need AT LEAST 2 common movies for the pearson algorithm
            continue
        sim = pearson_similarity(
            merged['rating_u'].values, merged['rating_v'].values)
        sims.append((other_id, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    top_neighbors = [uid for uid, s in sims[:N]]

    # "συνάρτηση πρόβλεψης σταθμισμένο μέσο όρο με αφαίρεση bias"
    pred_ratings = {}
    for movie_id in ratings_train['movieId'].unique():
        if movie_id in user_ratings['movieId'].values:
            continue  # ignores the already rated ones
        numerator = 0
        denominator = 0
        for neighbor_id in top_neighbors:
            neighbor_ratings = ratings_train[(ratings_train['userId'] == neighbor_id) &
                                             (ratings_train['movieId'] == movie_id)]
            if neighbor_ratings.empty:
                continue
            neighbor_mean = ratings_train[ratings_train['userId']
                                          == neighbor_id]['rating'].mean()
            sim = dict(sims).get(neighbor_id, 0)
            numerator += sim * \
                (neighbor_ratings.iloc[0]['rating'] - neighbor_mean)
            denominator += abs(sim)
        if denominator > 0:
            pred_ratings[movie_id] = user_mean + numerator / denominator

    # we inverse them to be in descending order and we pic the K highest rated ones
    top_movies = sorted(pred_ratings.items(),
                        key=lambda x: x[1], reverse=True)[:K]

    return pd.DataFrame(top_movies, columns=['movieId', 'score']).set_index('movieId')

# Σύστημα βάση περιεχομένου με βάση το cluster:


def recommend_cb_cluster(user_id, user_profile, cluster_profiles, movie_profiles, K):
    user_row = user_profile.loc[user_id]
    cluster_id = user_row['cluster']
    cluster_row = cluster_profiles.loc[cluster_id]

    # here we choose L movies to compare based on the cos similarity
    L = 3*K
    cluster_vec = cluster_row[all_genres].values.reshape(1, -1)
    movie_vecs = movie_profiles[all_genres].values
    sims = cosine_similarity(cluster_vec, movie_vecs).flatten()
    top_L_indices = sims.argsort()[::-1][:L]

    # final K movies that get recommened
    user_vec = user_row[all_genres].values.reshape(1, -1)
    subset_vecs = movie_vecs[top_L_indices]

    sims_user = cosine_similarity(
        user_vec, subset_vecs).flatten()

    top_K_local_indices = sims_user.argsort()[::-1][:K]
    top_K_global_indices = top_L_indices[top_K_local_indices]

    recs = movie_profiles.iloc[top_K_global_indices].copy()
    recs['score'] = sims_user[top_K_local_indices]
    return recs

# Σύστημα συνεργατικού φιλτραρίσματος με βάση το cluster


def recommend_cf_cluster(user_id, ratings_train, user_profile, N, K):
    user_row = user_profile.loc[user_id]
    cluster_id = user_row['cluster']

    cluster_users = user_profile[user_profile['cluster']
                                 == cluster_id].index.drop(user_id, errors='ignore')

    user_ratings = ratings_train[ratings_train['userId'] == user_id]
    user_mean = user_ratings['rating'].mean()

    sims = []
    for other_id in cluster_users:
        other_ratings = ratings_train[ratings_train['userId'] == other_id]
        merged = pd.merge(user_ratings, other_ratings,
                          on='movieId', suffixes=('_u', '_v'))
        if len(merged) < 2:
            continue
        sim = pearson_similarity(
            merged['rating_u'].values, merged['rating_v'].values)
        sims.append((other_id, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    top_neighbors = [uid for uid, s in sims[:N]]

    pred_ratings = {}
    for movie_id in ratings_train['movieId'].unique():
        if movie_id in user_ratings['movieId'].values:
            continue
        numerator = 0
        denominator = 0
        for neighbor_id in top_neighbors:
            neighbor_ratings = ratings_train[(ratings_train['userId'] == neighbor_id) &
                                             (ratings_train['movieId'] == movie_id)]
            if neighbor_ratings.empty:
                continue
            neighbor_mean = ratings_train[ratings_train['userId']
                                          == neighbor_id]['rating'].mean()
            sim = dict(sims).get(neighbor_id, 0)
            numerator += sim * \
                (neighbor_ratings.iloc[0]['rating'] - neighbor_mean)
            denominator += abs(sim)
        if denominator > 0:
            pred_ratings[movie_id] = user_mean + numerator / denominator

    top_movies = sorted(pred_ratings.items(),
                        key=lambda x: x[1], reverse=True)[:K]

    return pd.DataFrame(top_movies, columns=['movieId', 'score']).set_index('movieId')

# Question D
# here we calculate the MAE, Precision, Recall and Spearman Correlation


def calculate_metrics(user_id, recommendations, ratings_test, user_mean, K):
    if recommendations.empty:
        return None

    # movies in the test set for the user (the user_id)
    user_test_data = ratings_test[ratings_test['userId'] == user_id]
    if user_test_data.empty:
        return None

    # relevant movies in the test set aka where rating > user mean
    relevant_test_items = user_test_data[user_test_data['rating']
                                         > user_mean]['movieId'].values

    # top K recommended movies
    recommended_items = recommendations.index.tolist()  # movieId is index

    # Intersection
    hits = set(recommended_items).intersection(set(relevant_test_items))

    # Precision @ K
    precision = len(hits) / K

    # Recall @ K
    recall = len(hits) / \
        len(relevant_test_items) if len(relevant_test_items) > 0 else 0

    # MAE (only for the movies BOTH in the test set AND the recommendations)
    common_movies = recommendations.index.intersection(
        user_test_data['movieId'])
    mae = np.nan

    if len(common_movies) > 0:
        preds = recommendations.loc[common_movies, 'score']
        actuals = user_test_data.set_index(
            'movieId').loc[common_movies, 'rating']

        # here we calculate MAE only if the score is in the scale of a grade (aka >1, typically CF)
        # or if we know that it is CF
        if preds.mean() > 1.1:  # Heuristic: if it a rating (1-5) and not similarity (0-1)
            mae = np.mean(np.abs(preds - actuals))

    # Spearman Correlation
    spearman = np.nan

    # Top-K recommended movies
    topk_recs = recommendations.head(K)

    # Keep only movies that appear in test set
    user_test_data = user_test_data.set_index('movieId')
    common_movies = topk_recs.index.intersection(user_test_data.index)

    if len(common_movies) >= 2:
        # Predicted scores (order given by recommender)
        pred_scores = topk_recs.loc[common_movies, 'score']

        # Actual ratings from test set
        true_ratings = user_test_data.loc[common_movies, 'rating']

        # Convert to ranks
        pred_ranks = pred_scores.rank(ascending=False)
        true_ranks = true_ratings.rank(ascending=False)

        spearman, _ = stats.spearmanr(pred_ranks, true_ranks)

    return {"MAE": mae, "Precision": precision, "Recall": recall, "Spearman": spearman}

# Evaluation and Experiments


# Question A continues here
# we want to do this globaly (for every experiment) cause the genres are the same and do not change between experiments
movies["genres_list"] = movies["genres"].str.split("|")
all_genres = sorted(
    {genre for sublist in movies['genres_list'] for genre in sublist})

for genre in all_genres:
    movies[genre] = movies["genres"].apply(lambda x: 1 if genre in x else 0)

# this is basically the same as the normal movies, but here we simplify the table for the recommendation systems
# as to avoid ifs and comparissons when calculating the cos sim.
movie_profiles = movies.drop(
    ["genres", "genres_list"], axis=1).set_index('movieId')


def run_experiment(config_list, experiment_name):
    print(f"--- Starting {experiment_name} ---")
    results = []

    # Seeds for the 3 repetitions required
    seeds = [8, 6, 2]

    for config in config_list:
        split_ratio = config.get('split', 0.8)  # default 0.8 train
        N_val = config.get('N', 20)
        K_val = config.get('K', 10)

        print(f"Running: Split={split_ratio}, N={N_val}, K={K_val}...")

        seed_results = []

        for seed in seeds:
            # 1. Split Data
            r_train, r_test = train_test_split_per_user(
                ratings, test_size=(1-split_ratio), seed=seed)

            # 2. Create User Profiles (Question A continuation)
            # we re-create profiles here to avoid data leakage
            movies_exploded = movies.explode("genres_list")
            movies_exploded = movies_exploded.drop("genres", axis=1)

            ratings_merged = r_train.merge(movies_exploded.rename(
                columns={'genres_list': 'genres'}), on="movieId")

            user_profile = ratings_merged.groupby(["userId", "genres"])[
                "rating"].mean().unstack(fill_value=0)

            # ensure all genre columns exist for consistency
            for g in all_genres:
                if g not in user_profile.columns:
                    user_profile[g] = 0
            user_profile = user_profile[all_genres]

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
            user_profile['cluster_name'] = user_profile['cluster'].map(
                cluster_names)

            # 4. Run Evaluation (Question D continuation)
            metrics_acc = {"CB": [], "CF": [],
                           "CB_Cluster": [], "CF_Cluster": []}

            test_users = r_test['userId'].unique()

            for uid in test_users:
                u_mean = r_train[r_train['userId'] == uid]['rating'].mean()

                # Run Systems
                try:
                    recs = recommend_cb(
                        user_profile.loc[uid], movie_profiles, K_val)
                    m = calculate_metrics(uid, recs, r_test, u_mean, K_val)
                    if m:
                        metrics_acc["CB"].append(m)
                except KeyError:
                    pass

                try:
                    recs = recommend_cf(uid, r_train, N_val, K_val)
                    m = calculate_metrics(uid, recs, r_test, u_mean, K_val)
                    if m:
                        metrics_acc["CF"].append(m)
                except:
                    pass

                try:
                    recs = recommend_cb_cluster(
                        uid, user_profile, cluster_profiles, movie_profiles, K_val)
                    m = calculate_metrics(uid, recs, r_test, u_mean, K_val)
                    if m:
                        metrics_acc["CB_Cluster"].append(m)
                except KeyError:
                    pass

                try:
                    recs = recommend_cf_cluster(
                        uid, r_train, user_profile, N_val, K_val)
                    m = calculate_metrics(uid, recs, r_test, u_mean, K_val)
                    if m:
                        metrics_acc["CF_Cluster"].append(m)
                except:
                    pass

            # aggregate results for this seed
            for sys_name, m_list in metrics_acc.items():
                if m_list:
                    df_m = pd.DataFrame(m_list)
                    avg = df_m.mean(numeric_only=True).to_dict()
                    avg['System'] = sys_name
                    avg.update(config)
                    seed_results.append(avg)

        # aggregate results across the 3 seeds
        df_seeds = pd.DataFrame(seed_results)
        if not df_seeds.empty:
            group_cols = ['System'] + list(config.keys())
            df_final = df_seeds.groupby(group_cols).mean(
                numeric_only=True).reset_index()
            results.extend(df_final.to_dict('records'))

    return pd.DataFrame(results)

# Define Experiments


# Experiment 1: Split Ratios (N=20, K=10)
configs_exp1 = [
    {'split': 0.7, 'N': 20, 'K': 10},
    {'split': 0.8, 'N': 20, 'K': 10},
    {'split': 0.9, 'N': 20, 'K': 10}
]
df_exp1 = run_experiment(configs_exp1, "Experiment 1 (Splits)")

# Experiment 2: Neighbors (Split=80/20, K=10)
configs_exp2 = [
    {'split': 0.8, 'N': 10, 'K': 10},
    {'split': 0.8, 'N': 20, 'K': 10},
    {'split': 0.8, 'N': 50, 'K': 10}
]
df_exp2 = run_experiment(configs_exp2, "Experiment 2 (Neighbors)")

# Experiment 3: Top K (Split=80/20, N=20)
configs_exp3 = [
    {'split': 0.8, 'N': 20, 'K': 5},
    {'split': 0.8, 'N': 20, 'K': 10},
    {'split': 0.8, 'N': 20, 'K': 20}
]
df_exp3 = run_experiment(configs_exp3, "Experiment 3 (Top-K)")

# Export to Excel
print("Exporting results to 'experiment_results.xlsx'...")
with pd.ExcelWriter('experiment_results.xlsx') as writer:
    df_exp1.to_excel(writer, sheet_name='Exp1_Split', index=False)
    df_exp2.to_excel(writer, sheet_name='Exp2_Neighbors', index=False)
    df_exp3.to_excel(writer, sheet_name='Exp3_TopK', index=False)

print("Evaluation Complete.")
