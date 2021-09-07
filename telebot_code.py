import os
import telebot
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
        bot.send_message(user)
    except Exception as e:
        bot.reply_to(message, 'oooopse'+e)



# Enable saving next step handlers to file "./.handlers-saves/step.save".
# Delay=2 means that after any change in next step handlers (e.g. calling register_next_step_handler())
# saving will hapen after delay 2 seconds.
bot.enable_save_next_step_handlers(delay=2)
#@bot.message_handler(commands=['infos'])
#def infos(message):
#bot.send_message(message.chat.id,'Vos informations \n' +db['username'] +'\n '+ db['age'])


# Load next_step_handlers from save file (default "./.handlers-saves/step.save")
# WARNING It will work only if enable_save_next_step_handlers was called!
bot.load_next_step_handlers()

bot.polling()