from db_config import get_db_connection

connection = get_db_connection()

if connection:
    print("✅ MySQL connected successfully!")
    print("Database: visionx_ai")
    connection.close()
    print("Connection closed successfully.")
else:
    print("❌ MySQL connection failed.")