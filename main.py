import os
import telebot
import mysql.connector
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
        #db['username']=user.name
        # db['age']=user.age
        # db['taille']=user.taille
        # db['poids']=user.name
        # db['sexe']=user.sex
        # print(type(message.chat.id))
        # print(type(user.age))
        # print(type(user.sex))
        # print(type(user.name))
        # print(type(user.poids))
        # print(type(user.taille))
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

    @bot.message_handler(commands=['/food'])
    def food(message):
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



        bot.send_message(message.chat.id,'Votre profil est le suivant: \n' + 'Nom : ' + user["name"] + '\n Age : ' + str(user["age"]) + ' ans' + '\n Taille : ' + str(user["taille"]) + ' cm\n' + ' Poids : ' + str(user["poids"]) + ' Kg' + '\n Sexe : ' + user["sex"])
# Load next_step_handlers from save file (default "./.handlers-saves/step.save")
# WARNING It will work only if enable_save_next_step_handlers was called!
bot.load_next_step_handlers()

bot.polling()