import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
from db_config import get_db_connection

def display_users():
    conn = get_db_connection()
    if not conn:
        print("❌ Could not connect to MySQL database.")
        return

    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT user_id, username, email, role, is_active, created_at FROM users ORDER BY user_id ASC")
    users = cursor.fetchall()

    print("\n" + "=" * 80)
    print(f"{'ID':<5} | {'Username':<15} | {'Email':<25} | {'Role':<8} | {'Active':<6}")
    print("-" * 80)
    for u in users:
        active_str = "Yes" if u['is_active'] else "No"
        print(f"{u['user_id']:<5} | {u['username']:<15} | {u['email']:<25} | {u['role']:<8} | {active_str:<6}")
    print("=" * 80 + "\n")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    display_users()

