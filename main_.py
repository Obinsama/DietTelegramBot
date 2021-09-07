"""
alim_nom_fr: Collaborative Filtering for food Recommendations
Author: [Siddhartha Banerjee](https://twitter.com/sidd2006)
Date created: 2020/05/24
Last modified: 2020/05/24
Description: Recommending foods using a model trained on foodlens dataset.
"""
"""
## Introduction
This example demonstrates
[Collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)
using the [foodlens dataset](https://www.kaggle.com/c/foodlens-100k)
to recommend foods to users.
The foodLens ratings dataset lists the ratings given by a set of users to a set of foods.
Our goal is to be able to predict ratings for foods a user has not yet watched.
The foods with the highest predicted ratings can then be recommended to the user.
The steps in the model are as follows:
1. Map user ID to a "user vector" via an embedding matrix
2. Map food ID to a "food vector" via an embedding matrix
3. Compute the dot product between the user vector and food vector, to obtain
the a match score between the user and the food (predicted rating).
4. Train the embeddings via gradient descent using all known user-food pairs.
**References:**
- [Collaborative Filtering](https://dl.acm.org/doi/pdf/10.1145/371920.372071)
- [Neural Collaborative Filtering](https://dl.acm.org/doi/pdf/10.1145/3038912.3052569)
"""

import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt

"""
## First, load the data and apply preprocessing
"""

# Download the actual data from http://files.grouplens.org/datasets/foodlens/ml-latest-small.zip"
# Use the ratings.csv file
foodlens_data_file_url = (
    "http://files.grouplens.org/datasets/foodlens/ml-latest-small.zip"
)
foodlens_zipped_file = keras.utils.get_file(
    "ml-latest-small.zip", foodlens_data_file_url, extract=False
)
keras_datasets_path = Path(foodlens_zipped_file).parents[0]
print('keras_dataset_path',keras_datasets_path)
foodlens_dir = keras_datasets_path / "ml-latest-small"

# Only extract the data the first time the script is run.
if not foodlens_dir.exists():
    with ZipFile(foodlens_zipped_file, "r") as zip:
        # Extract files
        print("Extracting all the files now...")
        zip.extractall(path=keras_datasets_path)
        print("Done!")

#ratings_file = foodlens_dir / "ratings.csv"
ratings_file = "ratings.csv"

df = pd.read_csv(ratings_file)

"""
First, need to perform some preprocessing to encode users and foods as integer indices.
"""
user_ids = df["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
food_ids = df["foodId"].unique().tolist()
food2food_encoded = {x: i for i, x in enumerate(food_ids)}
food_encoded2food = {i: x for i, x in enumerate(food_ids)}
df["user"] = df["userId"].map(user2user_encoded)
df["food"] = df["foodId"].map(food2food_encoded)

num_users = len(user2user_encoded)
num_foods = len(food_encoded2food)
df["rating"] = df["rating"].values.astype(np.float32)
# min and max ratings will be used to normalize the ratings later
min_rating = min(df["rating"])
max_rating = max(df["rating"])

print(
    "Number of users: {}, Number of foods: {}, Min rating: {}, Max rating: {}".format(
        num_users, num_foods, min_rating, max_rating
    )
)

"""
## Prepare training and validation data
"""
df = df.sample(frac=1, random_state=42)
x = df[["user", "food"]].values
print('valeur de x',x)

# Normalize the targets between 0 and 1. Makes it easy to train.
y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
print('valeur de y',y)
# Assuming training on 90% of the data and validating on 10%.
train_indices = int(0.9 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:],
)

"""
## Create the model
We embed both users and foods in to 50-dimensional vectors.
The model computes a match score between user and food embeddings via a dot product,
and adds a per-food and per-user bias. The match score is scaled to the `[0, 1]`
interval via a sigmoid (since our ratings are normalized to this range).
"""
EMBEDDING_SIZE = 50


class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_foods, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_foods = num_foods
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.food_embedding = layers.Embedding(
            num_foods,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.food_bias = layers.Embedding(num_foods, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        food_vector = self.food_embedding(inputs[:, 1])
        food_bias = self.food_bias(inputs[:, 1])
        dot_user_food = tf.tensordot(user_vector, food_vector, 2)
        # Add all the components (including bias)
        x = dot_user_food + user_bias + food_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)


model = RecommenderNet(num_users, num_foods, EMBEDDING_SIZE)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001)
)

"""
## Train the model based on the data split
"""
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=64,
    epochs=50,
    verbose=1,
    validation_data=(x_val, y_val),
)

"""
## Plot training and validation loss
"""
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

"""
## Show top 10 food recommendations to a user
"""



#food_df = pd.read_csv(foodlens_dir / "foods.csv")
food_df=pd.read_csv("data.csv")
# Let us get a user and see the top recommendations.
user_id = df.userId.sample(1).iloc[0]
print('user_id',user_id)
foods_watched_by_user = df[df.userId == user_id]
foods_not_watched = food_df[
    ~food_df["foodId"].isin(foods_watched_by_user.foodId.values)
]["foodId"]
foods_not_watched = list(
    set(foods_not_watched).intersection(set(food2food_encoded.keys()))
)
foods_not_watched = [[food2food_encoded.get(x)] for x in foods_not_watched]
user_encoder = user2user_encoded.get(user_id)
user_food_array = np.hstack(
    ([[user_encoder]] * len(foods_not_watched), foods_not_watched)
)
ratings = model.predict(user_food_array).flatten()
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_food_ids = [
    food_encoded2food.get(foods_not_watched[x][0]) for x in top_ratings_indices
]

print("Showing recommendations for user: {}".format(user_id))
print("====" * 9)
print("foods with high ratings from user")
print("----" * 8)
top_foods_user = (
    foods_watched_by_user.sort_values(by="rating", ascending=False)
    .head(5)
    .foodId.values
)
food_df_rows = food_df[food_df["foodId"].isin(top_foods_user)]
for row in food_df_rows.itertuples():
    print(row.alim_nom_fr, ": Qte sucre", row.sucres)

print("----" * 8)
print("Top 10 food recommendations")
print("----" * 8)
recommended_foods = food_df[food_df["foodId"].isin(recommended_food_ids)]
for row in recommended_foods.itertuples():
    print(row.alim_nom_fr, ": Qte sucre", row.sucres)