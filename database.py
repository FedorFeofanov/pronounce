import psycopg2
from models import Word

def connect_to_db(db_name="pronounce", user="postgres", password="root", host='localhost', port=5432):
    """ Connect to a PostgreSQL database server """
    conn = None
    try:
        conn = psycopg2.connect(
            database=db_name,
            user=user,
            password=password,
            host=host,
            port=port
        )
        print("Connected to the PostgreSQL server successfully.")
        return conn
    except psycopg2.DatabaseError as e:
        print(f"Database error: {e}")
        return None
def execute_query(connection, query):
    if connection is None:
        return []

    try:
        cursor = connection.cursor()
        cursor.execute(query)
        connection.commit()
        records = []
        if cursor.description:
            records = cursor.fetchall()
        cursor.close()
        return records

    except psycopg2.DatabaseError as e:
        connection.rollback()
        print(f"Error during query execution: {e}")
        return []


def save_word(word: Word):
    connection = connect_to_db()
    if connection:
        data = execute_query(connection, 
                             f'''
                             insert into word(word, ipa_uk, audio_uk, ipa_us, audio_us) 
                             values($${word.word}$$, $${word.ipa_uk}$$, $${word.audio_uk}$$, $${word.ipa_us}$$, $${word.audio_us}$$);
                             ''')
        return data

def get_word_by_id(id):
    connection = connect_to_db()
    if(connection):
        data = execute_query(connection,
                             f'''
                             select id, word, ipa_uk, audio_uk, ipa_us, audio_us from word where id={id};
                            ''')[0]
        word = Word(data[0], data[1], data[2], data[3], data[4], data[5])
        return word

def get_max_id():
    connection = connect_to_db()
    if(connection):
        data = execute_query(connection, f'''
                                        select MAX(id) from word;
                                        ''')
def get_min_id():
    connection = connect_to_db()
    if(connection):
        data = execute_query(connection, f'''
                                        select MIN(id) from word;
                                        ''')