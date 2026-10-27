import os
import sys
from werkzeug.security import generate_password_hash

# Ensure backend directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from db_config import get_db_connection

def create_initial_admin(username, email, password):
    """
    Safely creates the initial system administrator if not already present.
    """
    if not username or not email or not password:
        print("[Error] Username, email, and password are required.")
        return False

    connection = get_db_connection()
    if not connection:
        print("[Error] Could not connect to MySQL database.")
        return False

    cursor = connection.cursor(dictionary=True)
    try:
        # Check if username or email exists
        cursor.execute("SELECT user_id, username, email FROM users WHERE username = %s OR email = %s", (username, email))
        existing = cursor.fetchone()
        if existing:
            print(f"[Warning] A user with username '{username}' or email '{email}' already exists.")
            return False

        pwd_hash = generate_password_hash(password)
        cursor.execute(
            """
            INSERT INTO users (username, email, password_hash, role, is_active)
            VALUES (%s, %s, %s, 'admin', 1)
            """,
            (username, email, pwd_hash)
        )
        connection.commit()
        admin_id = cursor.lastrowid
        print(f"[Success] Initial administrator '{username}' created successfully! (User ID: {admin_id})")
        return True

    except Exception as e:
        connection.rollback()
        print(f"[Error] Failed to create admin: {e}")
        return False
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    if len(sys.argv) == 4:
        u, e, p = sys.argv[1], sys.argv[2], sys.argv[3]
        create_initial_admin(u, e, p)
    else:
        print("Usage: python backend/create_admin.py <username> <email> <password>")
        print("Example: python backend/create_admin.py admin admin@visionx.clinic AdminPass123!")
