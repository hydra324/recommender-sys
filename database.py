import psycopg2
from config import config
import numpy as np

def get_user_feedback():
    '''
    Gets user-video feedback matrix from db.
    '''
    conn = None
    feedback_data = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        cur.execute("SELECT user_id,video_id,rating FROM user_liked")
        feedback_data = cur.fetchall()

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return np.array(feedback_data)

def get_random_k():
    conn = None
    count = 10
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM videos")
        count = cur.fetchone()

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return count

