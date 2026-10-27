import os
import mysql.connector
from mysql.connector import Error

def get_db_connection():
    """Establish and return a connection to MySQL using environment variables or defaults."""
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 3306)),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "1234"),
            database=os.getenv("DB_NAME", "visionx_ai")
        )

        if connection.is_connected():
            return connection

        return None

    except Error as e:
        print(f"MySQL connection error: {e}")
        return None