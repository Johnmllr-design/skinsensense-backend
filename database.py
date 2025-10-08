import sqlite3
import os

def init_db():
    connection = sqlite3.connect("database.skinsense.db")
    cursor = connection.cursor()

    cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                   username TEXT PRIMARY KEY,
                   password
                   )''')
    
    connection.commit()
    connection.close()


def login(username: str, password:str):
    connection = sqlite3.connect("database.skinsense.db")
    cursor = connection.cursor()

    cursor.execute("SELECT * FROM users WHERE password = ? AND username = ?", (password, username))     # select the password
    result = cursor.fetchone()
    print("fetching the provided profile is " + str(result))
    connection.close()
    return result is not None

def create_user(username:str, password:str, retyped_pw:str):
    if password != retyped_pw:
        return False
    print("the passwords! match")
    try:
        connection = sqlite3.connect("database.skinsense.db")
        cursor = connection.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        connection.commit()
        connection.close()
        print("succesfully created user " + str(username) + " with password " + str(password))
        return True
    except sqlite3.IntegrityError:
        print("username exists")
        return False
    finally:
        if connection:
            connection.close()

def print_datase():
    connection = sqlite3.connect("database.skinsense.db")
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users")
    rows = cursor.fetchall()
    print(rows)
    

        
