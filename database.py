import psycopg2
from models import Word, Phoneme


def connect_to_db(db_name="phonemes", user="postgres", password="root", host='localhost', port=5432):
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
def execute_query(connection, query, params=None):
    if connection is None:
        return []

    try:
        cursor = connection.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        connection.commit()
        records = []
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

def is_valid_phoneme(phoneme, word):
    connection = connect_to_db()
    if connection:
        query = f'''
            SELECT id 
            FROM phoneme_recordings
            WHERE word = %s AND phoneme LIKE %s
            LIMIT 1;
        '''
        phoneme_pattern = phoneme + "%"
        data_rows = execute_query(connection, query, (word, phoneme_pattern))
        if data_rows: return True
        print(f"The phoneme is {phoneme} the word is {word} and it is not in the db")
        return False

def get_phoneme_vector(phoneme, word, limit, sex, user_embedding):
    connection = connect_to_db()
    if connection:
        query = f'''
            SELECT id, word, phoneme, embedding, duration, sex, speaker_type, subset,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM phoneme_recordings 
            WHERE word = %s AND phoneme LIKE %s AND sex = %s
            ORDER BY similarity DESC 
            LIMIT %s;
        '''
        phoneme_pattern = phoneme + "%"
        data_rows = execute_query(connection, query, (user_embedding, word, phoneme_pattern, sex, limit))

        return data_rows

def is_valid_word(word, sex):
    connection = connect_to_db()
    if connection:
        query = f'''
            SELECT COUNT(*)
            FROM phoneme_recordings
            WHERE word = %s AND sex = %s;
        '''
        data_rows = execute_query(connection, query, (word, sex))
        if not data_rows or data_rows[0][0] < 10: return False
        return True