import os
import telebot
import mysql.connector
import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
from mysql.connector import Error

# Load TensorFlow Decision Forests

# Load the training dataset using pandas
import pandas
#import mysql.connector
from replit import db
from telebot import types

my_secret = os.getenv('API_KEY')
#my_secret ="1676961222:AAF7kZ_rf9olinM3ScqA2WqgdoqAo3APgws"
print(my_secret)
bot=telebot.TeleBot('1676961222:AAF7kZ_rf9olinM3ScqA2WqgdoqAo3APgws')
#@bot.message_handler(commands=['hello'])
#def hello(message):
#  bot.reply_to(message,'Bonjour pour utiliser l\'assistant veuillez entrer les informations suivantes:')
 # bot.send_message(message.chat.id,'nom(s) et prenom(s):')

#bot.polling()

#154.72.167.172
def insert(connection,message,ind):
    mySql_insert_query = """INSERT INTO Users (Chat_Id, Nom, Sexe,Age, Poids,Taille) 
                              VALUES 
                              (%s, %s,%s,%s,%s,%s) """
    record = (message.chat.id, ind.name, ind.sex, ind.age,ind.poids,ind.taille)
    cursor = connection.cursor()
    cursor.execute(mySql_insert_query,record)
    connection.commit()
    print(cursor.rowcount, "Record inserted successfully into Laptop table")
    cursor.close()
def create_table(connection):
    mySql_Create_Table_Query = """CREATE TABLE Users ( 
                                            Id int(11) NOT NULL,
                                            Chat_Id int(11) NOT NULL,
                                            Nom varchar(250) NOT NULL,
                                            Sexe varchar(250) NOT NULL,
                                            Age int(11) NOT NULL,
                                            Poids int(11) NOT NULL,
                                            Taille int(11) NOT NULL,
                                            PRIMARY KEY (Id)) """

    cursor = connection.cursor()
    result = cursor.execute(mySql_Create_Table_Query)
    # result = cursor.execute(mySql_Create_Table_Query)
    print("Laptop Table created successfully ")


def connect_sql():
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='tensor',
                                             user='root',
                                             password='')


    except mysql.connector.Error as error:
        print("Failed to create table in MySQL: {}".format(error))

    return connection
def close_conn(connection):
    if connection.is_connected():
        cursor = connection.cursor()
        cursor.close()
        connection.close()
        print("MySQL connection is closed")


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
print('keras_dataset_path', keras_datasets_path)
foodlens_dir = keras_datasets_path / "ml-latest-small"

# Only extract the data the first time the script is run.
if not foodlens_dir.exists():
    with ZipFile(foodlens_zipped_file, "r") as zip:
        # Extract files
        print("Extracting all the files now...")
        zip.extractall(path=keras_datasets_path)
        print("Done!")

# ratings_file = foodlens_dir / "ratings.csv"
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
#print('valeur de x', x)

# Normalize the targets between 0 and 1. Makes it easy to train.
y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
#print('valeur de y', y)
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



user_dict = {}


class User:
    def __init__(self, name):
        self.name = name
        self.age = None
        self.sex = None
        self.taille = None
        self.poids = None


# Handle '/start' and '/help'
@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    msg = bot.reply_to(message, """\
Bonjour, pour utiliser l'assistant, veuillez entrer vos parametres.
Quel est votre nom complet?
""")
    bot.register_next_step_handler(msg, process_name_step)


def process_name_step(message):
    try:
        chat_id = message.chat.id
        name = message.text
        user = User(name)
        user_dict[chat_id] = user
        msg = bot.reply_to(message, 'Quel est votre age?')
        bot.register_next_step_handler(msg, process_age_step)
    except Exception as e:
        bot.reply_to(message, 'oooops')


def process_age_step(message):
    try:
        chat_id = message.chat.id
        age = message.text
        if not age.isdigit():
            msg = bot.reply_to(message, 'Vous devez saisir un nombre quel est votre age?')
            bot.register_next_step_handler(msg, process_age_step)
            return
        user = user_dict[chat_id]
        user.age = age
        msg = bot.reply_to(message, 'Quelle est votre taille en cm?')
        bot.register_next_step_handler(msg, process_taille_step)
    except Exception as e:
        bot.reply_to(message, 'oooops')

def process_taille_step(message):
  try:
    chat_id=message.chat.id
    taille=message.text
    if not taille.isdigit():
       msg = bot.reply_to(message, 'Vous devez saisir un nombre quel est votre taille en cm?')
       bot.register_next_step_handler(msg, process_taille_step)
       return
    user=user_dict[chat_id]
    user.taille=taille
    msg = bot.reply_to(message, 'Quel est votre poids en Kg?')
    bot.register_next_step_handler(msg, process_poids_step)
  except Exception as e:
        bot.reply_to(message, 'oooops')

def process_poids_step(message):
  try:
    chat_id=message.chat.id
    poids=message.text
    if not poids.isdigit():
       msg = bot.reply_to(message, 'Vous devez saisir un nombre quel est votre poids en Kg?')
       bot.register_next_step_handler(msg, process_poids_step)
       return
    user=user_dict[chat_id]
    user.poids=poids
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
    markup.add('Homme', 'Femme')
    msg = bot.reply_to(message, 'Quel est votre genre?', reply_markup=markup)
    bot.register_next_step_handler(msg, process_sex_step)
  except Exception as e:
        bot.reply_to(message, 'oooops')

def process_sex_step(message):
    try:
        chat_id = message.chat.id
        sex = message.text
        user = user_dict[chat_id]
        if (sex == u'Homme') or (sex == u'Femme'):
            user.sex = sex
        else:
            raise Exception("sexe inconnu")
        bot.send_message(chat_id, 'Merci d\'avoir suivi la procedure votre profil est le suivant: \n' 'Nom : ' + user.name + '\n Age : ' + str(user.age)+' ans' +'\n Taille : ' + user.taille+' cm' +'\n Poids : ' + user.poids+' Kg' +'\n Sexe : ' + user.sex)

        con=connect_sql()
        insert(con,message,user)
        close_conn(con)
        #bot.send_message(user)
    except Exception as e:
        bot.reply_to(message, 'oooopse'+e)



# Enable saving next step handlers to file "./.handlers-saves/step.save".
# Delay=2 means that after any change in next step handlers (e.g. calling register_next_step_handler())
# saving will hapen after delay 2 seconds.
bot.enable_save_next_step_handlers(delay=2)
@bot.message_handler(commands=['infos'])
def infos(message):
    chat_id=message.chat.id
    #user = user_dict[chat_id]
    connection = connect_sql()
    cursor = connection.cursor()
    sql_select_Query = """select * from Users where Chat_Id = %s"""
    #record=message.chat.id
    cursor.execute(sql_select_Query,(message.chat.id,))
    # get all records
    records = cursor.fetchall()
    print("Total number of rows in table: ", cursor.rowcount)
    print("\nSending it to telegram")
    for row in records:
        user={
            "name":row[2],
            "age":row[4],
            "sex":row[3],
            "poids":row[5],
            "taille":row[6]
        }

        # print("Id = ", row[0], )
        # print("Chat_id = ", row[1])
        # print("name  = ", row[2])
        # print("age  = ", row[3])
        # print("sex  = ", row[4])
        # print("poids  = ", row[5])
        # print("taille  = ", row[6])
        # print("Purchase date  = ", row[3], "\n")
    bot.send_message(message.chat.id,'Votre profil est le suivant: \n' +'Nom : ' + user["name"] + '\n Age : ' + str(user["age"]) + ' ans' + '\n Taille : ' + str(user["taille"]) + ' cm\n' + ' Poids : ' + str(user["poids"]) + ' Kg' + '\n Sexe : ' + user["sex"])

@bot.message_handler(commands=['food'])
def food(message):
    print("Bonjour")
    chat_id = message.chat.id
    # user = user_dict[chat_id]
    connection = connect_sql()
    cursor = connection.cursor()
    sql_select_Query = """select * from Users where Chat_Id = %s"""
    # record=message.chat.id
    cursor.execute(sql_select_Query, (message.chat.id,))
    # get all records
    records = cursor.fetchall()
    print("Total number of rows in table: ", cursor.rowcount)
    print("\nSending it to telegram")
    for row in records:
        user = {
            "name": row[2],
            "age": row[4],
            "sex": row[3],
            "poids": row[5],
            "taille": row[6]
        }
    if user['age']>30 & user['age']<50 & user['poids']>60 & user['poids']<100:
        userId_=1
    elif user['age']>10 & user['age']<20 & user['poids']>60 & user['poids']<100:
        userId_=5
    elif user['age']>50 & user['age']<60 & user['poids']>60 & user['poids']<100:
        userId_=3
    elif user['age']>20 & user['age']<30 & user['poids']>60 & user['poids']<100:
        userId_=9
    elif user['age']>60 & user['age']<70 & user['poids']>60 & user['poids']<100:
        userId_=11
    elif user['age']>70 & user['age']<100 & user['poids']>60 & user['poids']<100:
        userId_=15
    else:
        userId_=10




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

    # food_df = pd.read_csv(foodlens_dir / "foods.csv")
    food_df = pd.read_csv("data.csv")
    # Let us get a user and see the top recommendations.
    #user_id = df.userId.sample(1).iloc[0]
    user_id=userId_
    print('user_id', user_id)
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

    bot.send_message(message.chat.id,"Showing recommendations for user: {}".format(user_id))
    bot.send_message(message.chat.id,"====" * 12)
    bot.send_message(message.chat.id,"foods with high ratings from ciqual")
    bot.send_message(message.chat.id,"----" * 20)

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
    rec=""
    for row in food_df_rows.itertuples():
        print(row.alim_nom_fr, ": Qte sucre", row.sucres)
        rec+="\n"+"----"+"\n"+row.alim_nom_fr+ " : Qte sucre "+ str(row.sucres)
    bot.send_message(message.chat.id, rec)


    bot.send_message(message.chat.id,"----" * 20)
    bot.send_message(message.chat.id,"Top 10 food recommendations")
    bot.send_message(message.chat.id,"----" * 20)

    print("----" * 8)
    print("Top 10 food recommendations")
    print("----" * 8)
    recommended_foods = food_df[food_df["foodId"].isin(recommended_food_ids)]
    eat=""
    for row in recommended_foods.itertuples():
        print(row.alim_nom_fr, ": qte glucides", row.glucides,"qte sucre",row.sucres,'qte eau',row.eau)
        eat+="\n"+"----------------------------------------------------------------------------------------"+"\n"+row.alim_nom_fr+ " : Qte sucre "+str(row.sucres)
    bot.send_message(message.chat.id,eat)

    #bot.send_message(message.chat.id,'Votre profil est le suivant: \n' + 'Nom : ' + user["name"] + '\n Age : ' + str(user["age"]) + ' ans' + '\n Taille : ' + str(user["taille"]) + ' cm\n' + ' Poids : ' + str(user["poids"]) + ' Kg' + '\n Sexe : ' + user["sex"])
# Load next_step_handlers from save file (default "./.handlers-saves/step.save")
# WARNING It will work only if enable_save_next_step_handlers was called!
bot.load_next_step_handlers()

bot.polling()