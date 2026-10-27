import os
from werkzeug.security import generate_password_hash, check_password_hash
from db_config import get_db_connection

# =========================================================================
# PATIENT OPERATIONS (PRESERVED & EXTENDED)
# =========================================================================

def save_patient(patient_name, age, place):
    """
    Save patient details in MySQL and return the generated patient ID.
    Preserves existing functionality.
    """
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return None

        cursor = connection.cursor()
        query = "INSERT INTO patients (patient_name, age, place) VALUES (%s, %s, %s)"
        cursor.execute(query, (patient_name, age, place))
        patient_id = cursor.lastrowid
        connection.commit()
        return patient_id
    except Exception as error:
        if connection is not None:
            connection.rollback()
        print(f"Error saving patient: {error}")
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def get_patient(patient_id):
    """Retrieve patient details using the patient ID."""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return None

        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT patient_id, patient_name, age, place, created_at
            FROM patients
            WHERE patient_id = %s
        """
        cursor.execute(query, (patient_id,))
        return cursor.fetchone()
    except Exception as error:
        print(f"Error retrieving patient: {error}")
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def search_patients(search_term=""):
    """Search for existing patients by name, place, or ID."""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return []

        cursor = connection.cursor(dictionary=True)
        term = f"%{search_term.strip()}%"
        query = """
            SELECT patient_id, patient_name, age, place, created_at
            FROM patients
            WHERE patient_name LIKE %s OR place LIKE %s OR CAST(patient_id AS CHAR) LIKE %s
            ORDER BY patient_id DESC
            LIMIT 50
        """
        cursor.execute(query, (term, term, term))
        return cursor.fetchall()
    except Exception as error:
        print(f"Error searching patients: {error}")
        return []
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def get_all_patients(limit=100):
    """Retrieve list of all registered patients."""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return []

        cursor = connection.cursor(dictionary=True)
        query = "SELECT patient_id, patient_name, age, place, created_at FROM patients ORDER BY patient_id DESC LIMIT %s"
        cursor.execute(query, (limit,))
        return cursor.fetchall()
    except Exception as error:
        print(f"Error retrieving all patients: {error}")
        return []
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


# =========================================================================
# USER & AUTHENTICATION OPERATIONS
# =========================================================================

def get_user_by_username_or_email(identifier):
    """Find a user record by username or email."""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return None

        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT user_id, username, email, password_hash, role, is_active, created_at
            FROM users
            WHERE username = %s OR email = %s
        """
        cursor.execute(query, (identifier, identifier))
        return cursor.fetchone()
    except Exception as error:
        print(f"Error getting user by identifier: {error}")
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def get_user_by_id(user_id):
    """Find a user record by primary key user_id."""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return None

        cursor = connection.cursor(dictionary=True)
        query = "SELECT user_id, username, email, role, is_active, created_at FROM users WHERE user_id = %s"
        cursor.execute(query, (user_id,))
        return cursor.fetchone()
    except Exception as error:
        print(f"Error getting user by ID: {error}")
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def create_user(username, email, plain_password, role, is_active=1):
    """Create a new user account with hashed password."""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return None

        cursor = connection.cursor()
        hashed_pw = generate_password_hash(plain_password)
        query = """
            INSERT INTO users (username, email, password_hash, role, is_active)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (username, email, hashed_pw, role, is_active))
        user_id = cursor.lastrowid
        connection.commit()
        return user_id
    except Exception as error:
        if connection is not None:
            connection.rollback()
        print(f"Error creating user: {error}")
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def toggle_user_active_status(user_id, is_active):
    """Activate or deactivate a user account."""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return False

        cursor = connection.cursor()
        query = "UPDATE users SET is_active = %s WHERE user_id = %s"
        cursor.execute(query, (is_active, user_id))
        connection.commit()
        return cursor.rowcount > 0
    except Exception as error:
        if connection is not None:
            connection.rollback()
        print(f"Error toggling user status: {error}")
        return False
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def get_all_users():
    """Retrieve all user accounts (omitting password hash)."""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return []

        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT u.user_id, u.username, u.email, u.role, u.is_active, u.created_at,
                   d.doctor_id, d.full_name as doctor_name, d.specialization
            FROM users u
            LEFT JOIN doctors d ON u.user_id = d.user_id
            ORDER BY u.user_id DESC
        """
        cursor.execute(query)
        return cursor.fetchall()
    except Exception as error:
        print(f"Error retrieving all users: {error}")
        return []
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


# =========================================================================
# DOCTOR OPERATIONS
# =========================================================================

def create_doctor_account(username, email, plain_password, full_name, specialization, phone=""):
    """
    Atomically creates a user account with role 'doctor' and the linked doctor profile.
    """
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return None

        cursor = connection.cursor()
        # 1. Insert into users
        hashed_pw = generate_password_hash(plain_password)
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role, is_active) VALUES (%s, %s, %s, 'doctor', 1)",
            (username, email, hashed_pw)
        )
        user_id = cursor.lastrowid

        # 2. Insert into doctors
        cursor.execute(
            "INSERT INTO doctors (user_id, full_name, specialization, phone) VALUES (%s, %s, %s, %s)",
            (user_id, full_name, specialization, phone)
        )
        doctor_id = cursor.lastrowid

        connection.commit()
        return doctor_id
    except Exception as error:
        if connection is not None:
            connection.rollback()
        print(f"Error creating doctor account: {error}")
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def get_doctor_by_user_id(user_id):
    """Retrieve doctor profile linked to user_id."""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return None

        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT d.doctor_id, d.user_id, d.full_name, d.specialization, d.phone, d.created_at,
                   u.username, u.email, u.is_active
            FROM doctors d
            JOIN users u ON d.user_id = u.user_id
            WHERE d.user_id = %s
        """
        cursor.execute(query, (user_id,))
        return cursor.fetchone()
    except Exception as error:
        print(f"Error retrieving doctor by user ID: {error}")
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def get_doctor_by_id(doctor_id):
    """Retrieve doctor profile by doctor_id."""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return None

        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT d.doctor_id, d.user_id, d.full_name, d.specialization, d.phone, d.created_at,
                   u.username, u.email, u.is_active
            FROM doctors d
            JOIN users u ON d.user_id = u.user_id
            WHERE d.doctor_id = %s
        """
        cursor.execute(query, (doctor_id,))
        return cursor.fetchone()
    except Exception as error:
        print(f"Error retrieving doctor by doctor ID: {error}")
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def get_active_doctors():
    """Retrieve list of all active doctors for staff assignment."""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return []

        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT d.doctor_id, d.user_id, d.full_name, d.specialization, d.phone
            FROM doctors d
            JOIN users u ON d.user_id = u.user_id
            WHERE u.is_active = 1
            ORDER BY d.full_name ASC
        """
        cursor.execute(query)
        return cursor.fetchall()
    except Exception as error:
        print(f"Error retrieving active doctors: {error}")
        return []
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


# =========================================================================
# EXAMINATION OPERATIONS
# =========================================================================

def create_examination(patient_id, doctor_id, staff_user_id, notes=None):
    """
    Creates an examination linking patient, assigned doctor, and staff member.
    """
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return None

        cursor = connection.cursor()
        query = """
            INSERT INTO examinations (patient_id, doctor_id, staff_user_id, notes)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (patient_id, doctor_id, staff_user_id, notes))
        examination_id = cursor.lastrowid
        connection.commit()
        return examination_id
    except Exception as error:
        if connection is not None:
            connection.rollback()
        print(f"Error creating examination: {error}")
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def get_examination_details(examination_id):
    """
    Retrieves full details of an examination including patient, doctor, staff,
    image, and AI prediction.
    """
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return None

        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT e.examination_id, e.patient_id, e.doctor_id, e.staff_user_id, e.examined_at, e.notes,
                   p.patient_name, p.age as patient_age, p.place as patient_place,
                   d.full_name as doctor_name, d.specialization as doctor_specialization,
                   u_staff.username as staff_username,
                   img.image_id, img.image_name, img.image_path,
                   pred.prediction_id, pred.model_version, pred.detected_disease,
                   pred.confidence_score, pred.heatmap_url, pred.report_path, pred.processed_at
            FROM examinations e
            JOIN patients p ON e.patient_id = p.patient_id
            JOIN doctors d ON e.doctor_id = d.doctor_id
            LEFT JOIN users u_staff ON e.staff_user_id = u_staff.user_id
            LEFT JOIN eye_images img ON e.examination_id = img.examination_id
            LEFT JOIN ai_predictions pred ON img.image_id = pred.image_id
            WHERE e.examination_id = %s
        """
        cursor.execute(query, (examination_id,))
        return cursor.fetchone()
    except Exception as error:
        print(f"Error retrieving examination details: {error}")
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def get_examinations_for_doctor(doctor_id, limit=50):
    """Retrieve all examinations assigned to a specific doctor."""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return []

        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT e.examination_id, e.patient_id, e.doctor_id, e.staff_user_id, e.examined_at, e.notes,
                   p.patient_name, p.age as patient_age, p.place as patient_place,
                   img.image_id, img.image_name, img.image_path,
                   pred.prediction_id, pred.detected_disease, pred.confidence_score, pred.report_path
            FROM examinations e
            JOIN patients p ON e.patient_id = p.patient_id
            LEFT JOIN eye_images img ON e.examination_id = img.examination_id
            LEFT JOIN ai_predictions pred ON img.image_id = pred.image_id
            WHERE e.doctor_id = %s
            ORDER BY e.examination_id DESC
            LIMIT %s
        """
        cursor.execute(query, (doctor_id, limit))
        return cursor.fetchall()
    except Exception as error:
        print(f"Error retrieving doctor examinations: {error}")
        return []
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def get_examinations_for_staff(staff_user_id, limit=50):
    """Retrieve examinations processed by a specific staff member."""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return []

        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT e.examination_id, e.patient_id, e.doctor_id, e.staff_user_id, e.examined_at, e.notes,
                   p.patient_name, p.age as patient_age, p.place as patient_place,
                   d.full_name as doctor_name,
                   img.image_id, img.image_name, img.image_path,
                   pred.prediction_id, pred.detected_disease, pred.confidence_score, pred.report_path
            FROM examinations e
            JOIN patients p ON e.patient_id = p.patient_id
            JOIN doctors d ON e.doctor_id = d.doctor_id
            LEFT JOIN eye_images img ON e.examination_id = img.examination_id
            LEFT JOIN ai_predictions pred ON img.image_id = pred.image_id
            WHERE e.staff_user_id = %s
            ORDER BY e.examination_id DESC
            LIMIT %s
        """
        cursor.execute(query, (staff_user_id, limit))
        return cursor.fetchall()
    except Exception as error:
        print(f"Error retrieving staff examinations: {error}")
        return []
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def get_all_examinations(limit=100):
    """Retrieve all examinations for admin overview."""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return []

        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT e.examination_id, e.patient_id, e.doctor_id, e.staff_user_id, e.examined_at, e.notes,
                   p.patient_name, d.full_name as doctor_name, u.username as staff_username,
                   pred.prediction_id, pred.detected_disease, pred.confidence_score, pred.report_path
            FROM examinations e
            JOIN patients p ON e.patient_id = p.patient_id
            JOIN doctors d ON e.doctor_id = d.doctor_id
            LEFT JOIN users u ON e.staff_user_id = u.user_id
            LEFT JOIN eye_images img ON e.examination_id = img.examination_id
            LEFT JOIN ai_predictions pred ON img.image_id = pred.image_id
            ORDER BY e.examination_id DESC
            LIMIT %s
        """
        cursor.execute(query, (limit,))
        return cursor.fetchall()
    except Exception as error:
        print(f"Error retrieving all examinations: {error}")
        return []
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


# =========================================================================
# PREDICTION & IMAGE OPERATIONS (PRESERVED & EXTENDED)
# =========================================================================

def save_prediction(
    image_name,
    image_path,
    detected_disease,
    confidence_score,
    model_version,
    heatmap_url=None,
    report_path=None,
    patient_id=None,
    examination_id=None
):
    """
    Save uploaded image details and AI prediction to MySQL.
    Preserves existing functionality while supporting examination_id.
    """
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return None

        cursor = connection.cursor()

        # 1. Save uploaded image details, linked to patient and examination
        image_query = """
            INSERT INTO eye_images (
                image_name,
                image_path,
                patient_id,
                examination_id
            )
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(image_query, (image_name, image_path, patient_id, examination_id))
        image_id = cursor.lastrowid

        # 2. Save prediction details
        prediction_query = """
            INSERT INTO ai_predictions (
                image_id,
                model_version,
                detected_disease,
                confidence_score,
                heatmap_url,
                report_path
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        confidence_fraction = float(confidence_score) / 100.0
        cursor.execute(
            prediction_query,
            (
                image_id,
                model_version,
                detected_disease,
                confidence_fraction,
                heatmap_url,
                report_path
            )
        )
        prediction_id = cursor.lastrowid
        connection.commit()
        return prediction_id
    except Exception as error:
        if connection is not None:
            connection.rollback()
        print(f"Error saving prediction: {error}")
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def update_report_path(prediction_id, report_path):
    """Update an existing prediction record with its generated PDF path."""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return False

        cursor = connection.cursor()
        query = "UPDATE ai_predictions SET report_path = %s WHERE prediction_id = %s"
        cursor.execute(query, (report_path, prediction_id))
        connection.commit()
        return cursor.rowcount > 0
    except Exception as error:
        if connection is not None:
            connection.rollback()
        print(f"Error updating report path: {error}")
        return False
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def get_report_path(prediction_id):
    """Retrieve the saved PDF report path from MySQL using the prediction ID."""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return None

        cursor = connection.cursor()
        query = "SELECT report_path FROM ai_predictions WHERE prediction_id = %s"
        cursor.execute(query, (prediction_id,))
        result = cursor.fetchone()
        return result[0] if (result and result[0]) else None
    except Exception as error:
        print(f"Error retrieving PDF report path: {error}")
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def get_prediction_by_id(prediction_id):
    """Retrieve full prediction record with linked patient and examination info."""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if connection is None:
            return None

        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT p.prediction_id, p.image_id, p.model_version, p.detected_disease,
                   p.confidence_score, p.heatmap_url, p.report_path, p.processed_at,
                   img.image_name, img.image_path, img.patient_id, img.examination_id,
                   e.doctor_id, e.staff_user_id
            FROM ai_predictions p
            JOIN eye_images img ON p.image_id = img.image_id
            LEFT JOIN examinations e ON img.examination_id = e.examination_id
            WHERE p.prediction_id = %s
        """
        cursor.execute(query, (prediction_id,))
        return cursor.fetchone()
    except Exception as error:
        print(f"Error retrieving prediction: {error}")
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


# =========================================================================
# DASHBOARD STATS
# =========================================================================

def get_admin_dashboard_stats():
    """Retrieve aggregate counts for the Admin Dashboard."""
    connection = None
    cursor = None
    stats = {
        "total_patients": 0,
        "total_doctors": 0,
        "total_staff": 0,
        "total_examinations": 0,
        "total_predictions": 0,
    }
    try:
        connection = get_db_connection()
        if connection is None:
            return stats

        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM patients")
        stats["total_patients"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM doctors")
        stats["total_doctors"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'staff'")
        stats["total_staff"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM examinations")
        stats["total_examinations"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM ai_predictions")
        stats["total_predictions"] = cursor.fetchone()[0]

        return stats
    except Exception as error:
        print(f"Error fetching admin stats: {error}")
        return stats
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()