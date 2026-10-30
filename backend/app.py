import os
import sys
import uuid
import re
from functools import wraps

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory,
    send_file,
    abort,
    session,
    redirect,
    url_for,
    flash,
)

from PIL import Image
from werkzeug.security import check_password_hash

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

project_root = os.path.dirname(current_dir)
frontend_dir = os.path.join(project_root, "frontend")
template_dir = os.path.join(frontend_dir, "templates")
static_dir = os.path.join(frontend_dir, "static")

UPLOAD_DIR = os.path.join(current_dir, "uploads")
REPORT_DIR = os.path.join(current_dir, "reports")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# --------------------------------------------------
# PROJECT MODULES
# --------------------------------------------------

from model_loader import load_model, get_model
from prediction import predict_with_details, validate_fundus_image
from gradcam import generate_gradcam, DEFAULT_GENERATED_DIR

from db_operations import (
    save_patient,
    get_patient,
    search_patients,
    get_all_patients,
    save_prediction,
    update_report_path,
    get_report_path,
    get_user_by_username_or_email,
    get_user_by_id,
    create_user,
    toggle_user_active_status,
    get_all_users,
    create_doctor_account,
    get_doctor_by_user_id,
    get_doctor_by_id,
    get_active_doctors,
    create_examination,
    get_examination_details,
    get_examinations_for_doctor,
    get_examinations_for_staff,
    get_all_examinations,
    get_prediction_by_id,
    get_admin_dashboard_stats,
    get_patient_examinations_history,
)

from pdf_report import generate_pdf_report


# --------------------------------------------------
# FLASK APPLICATION
# --------------------------------------------------

app = Flask(
    __name__,
    template_folder=template_dir,
    static_folder=static_dir,
)

# Secure Session Configuration
app.secret_key = os.getenv("FLASK_SECRET_KEY", "visionx_ai_production_secret_key_2026_mca")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Limit uploads to 15 MB
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024


# --------------------------------------------------
# PRELOAD MODEL
# --------------------------------------------------

try:
    print("[Flask] Initializing model loading...")
    load_model()
    print("[Flask] Model successfully loaded.")
except Exception as error:
    print(f"[Flask Warning] Model preload failed: {error}")


# --------------------------------------------------
# AUTHENTICATION & ACCESS CONTROL DECORATORS
# --------------------------------------------------

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            if request.is_json:
                return jsonify({"success": False, "error": "Authentication required."}), 401
            return redirect(url_for("login_page", next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def role_required(*allowed_roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if "user_id" not in session:
                if request.is_json:
                    return jsonify({"success": False, "error": "Authentication required."}), 401
                return redirect(url_for("login_page", next=request.url))

            user_role = session.get("role")
            if user_role not in allowed_roles:
                if request.is_json:
                    return jsonify({"success": False, "error": "Unauthorized access for your role."}), 403
                return render_template("403.html", message="You do not have permission to access this portal."), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator


# --------------------------------------------------
# PUBLIC & AUTH ROUTES
# --------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    # If user is already authenticated, redirect to their respective dashboard
    if "user_id" in session:
        role = session.get("role")
        if role == "admin":
            return redirect(url_for("admin_dashboard"))
        elif role == "staff":
            return redirect(url_for("staff_dashboard"))
        elif role == "doctor":
            return redirect(url_for("doctor_dashboard"))

    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "GET":
        if "user_id" in session:
            role = session.get("role")
            if role == "admin":
                return redirect(url_for("admin_dashboard"))
            elif role == "staff":
                return redirect(url_for("staff_dashboard"))
            elif role == "doctor":
                return redirect(url_for("doctor_dashboard"))
        return render_template("login.html")

    # POST - Handle Login Request
    data = request.get_json(silent=True) or request.form
    identifier = str(data.get("identifier", "")).strip()
    password = str(data.get("password", "")).strip()
    expected_role = data.get("role")

    if not identifier or not password:
        msg = "Please provide username/email and password."
        if request.is_json:
            return jsonify({"success": False, "error": msg}), 400
        flash(msg, "error")
        return render_template("login.html"), 400

    user = get_user_by_username_or_email(identifier)
    if not user:
        msg = "Invalid username or password."
        if request.is_json:
            return jsonify({"success": False, "error": msg}), 401
        flash(msg, "error")
        return render_template("login.html"), 401

    if not user.get("is_active"):
        msg = "This account has been deactivated. Please contact an administrator."
        if request.is_json:
            return jsonify({"success": False, "error": msg}), 403
        flash(msg, "error")
        return render_template("login.html"), 403

    if expected_role and user.get("role") != expected_role:
        msg = f"Account exists, but role is '{user.get('role')}', not '{expected_role}'."
        if request.is_json:
            return jsonify({"success": False, "error": msg}), 403
        flash(msg, "error")
        return render_template("login.html"), 403

    if not check_password_hash(user["password_hash"], password):
        msg = "Invalid username or password."
        if request.is_json:
            return jsonify({"success": False, "error": msg}), 401
        flash(msg, "error")
        return render_template("login.html"), 401

    # Establish authenticated session
    session.clear()
    session["user_id"] = user["user_id"]
    session["username"] = user["username"]
    session["email"] = user["email"]
    session["role"] = user["role"]

    # If role is doctor, cache doctor_id
    if user["role"] == "doctor":
        doc = get_doctor_by_user_id(user["user_id"])
        if doc:
            session["doctor_id"] = doc["doctor_id"]
            session["full_name"] = doc["full_name"]

    redirect_url = url_for("admin_dashboard") if user["role"] == "admin" else (
        url_for("staff_dashboard") if user["role"] == "staff" else url_for("doctor_dashboard")
    )

    if request.is_json:
        return jsonify({
            "success": True,
            "message": "Login successful.",
            "redirect_url": redirect_url,
            "role": user["role"]
        }), 200

    return redirect(redirect_url)


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("login_page"))


# --------------------------------------------------
# PORTAL 1: ADMIN PORTAL
# --------------------------------------------------

@app.route("/admin/dashboard", methods=["GET"])
@role_required("admin")
def admin_dashboard():
    stats = get_admin_dashboard_stats()
    users = get_all_users()
    recent_examinations = get_all_examinations(limit=10)
    return render_template(
        "admin/dashboard.html",
        stats=stats,
        users=users,
        recent_examinations=recent_examinations
    )


@app.route("/admin/users", methods=["GET"])
@role_required("admin")
def admin_users_page():
    users = get_all_users()
    return render_template("admin/users.html", users=users)


@app.route("/admin/create-staff", methods=["POST"])
@role_required("admin")
def admin_create_staff():
    data = request.get_json(silent=True) or request.form
    username = str(data.get("username", "")).strip()
    email = str(data.get("email", "")).strip()
    password = str(data.get("password", "")).strip()

    if not username or not email or not password:
        return jsonify({"success": False, "error": "Username, email, and password are required."}), 400

    if len(password) < 6:
        return jsonify({"success": False, "error": "Password must be at least 6 characters."}), 400

    existing = get_user_by_username_or_email(username) or get_user_by_username_or_email(email)
    if existing:
        return jsonify({"success": False, "error": "A user with this username or email already exists."}), 400

    user_id = create_user(username, email, password, "staff")
    if not user_id:
        return jsonify({"success": False, "error": "Failed to create hospital staff account."}), 500

    return jsonify({"success": True, "message": f"Staff account '{username}' created successfully."}), 201


@app.route("/admin/create-admin", methods=["POST"])
@role_required("admin")
def admin_create_admin():
    data = request.get_json(silent=True) or request.form
    username = str(data.get("username", "")).strip()
    email = str(data.get("email", "")).strip()
    password = str(data.get("password", "")).strip()

    if not username or not email or not password:
        return jsonify({"success": False, "error": "All fields are required."}), 400

    if len(password) < 8:
        return jsonify({"success": False, "error": "Admin password must be at least 8 characters."}), 400

    existing = get_user_by_username_or_email(username) or get_user_by_username_or_email(email)
    if existing:
        return jsonify({"success": False, "error": "A user with this username or email already exists."}), 400

    user_id = create_user(username, email, password, "admin")
    if not user_id:
        return jsonify({"success": False, "error": "Failed to create admin account."}), 500

    return jsonify({"success": True, "message": f"Admin account '{username}' created successfully."}), 201


@app.route("/admin/create-doctor", methods=["POST"])
@role_required("admin")
def admin_create_doctor():
    data = request.get_json(silent=True) or request.form
    username = str(data.get("username", "")).strip()
    email = str(data.get("email", "")).strip()
    password = str(data.get("password", "")).strip()
    full_name = str(data.get("full_name", "")).strip()
    specialization = str(data.get("specialization", "")).strip() or "Ophthalmologist"
    phone = str(data.get("phone", "")).strip()

    if not username or not email or not password or not full_name:
        return jsonify({"success": False, "error": "Username, email, password, and doctor full name are required."}), 400

    existing = get_user_by_username_or_email(username) or get_user_by_username_or_email(email)
    if existing:
        return jsonify({"success": False, "error": "A user with this username or email already exists."}), 400

    doctor_id = create_doctor_account(username, email, password, full_name, specialization, phone)
    if not doctor_id:
        return jsonify({"success": False, "error": "Failed to create doctor account."}), 500

    return jsonify({"success": True, "message": f"Doctor profile for 'Dr. {full_name}' created successfully."}), 201


@app.route("/admin/toggle-user-status", methods=["POST"])
@role_required("admin")
def admin_toggle_user_status():
    data = request.get_json(silent=True) or request.form
    target_user_id = data.get("user_id")
    is_active = data.get("is_active")

    if target_user_id is None or is_active is None:
        return jsonify({"success": False, "error": "User ID and status are required."}), 400

    target_user_id = int(target_user_id)
    is_active = 1 if int(is_active) == 1 else 0

    # Prevent admin self-lockout
    if target_user_id == session.get("user_id"):
        return jsonify({"success": False, "error": "Security violation: You cannot deactivate your own admin account."}), 400

    success = toggle_user_active_status(target_user_id, is_active)
    if success:
        action_name = "activated" if is_active else "deactivated"
        return jsonify({"success": True, "message": f"User account has been {action_name}."}), 200
    return jsonify({"success": False, "error": "Failed to update account status."}), 500


@app.route("/admin/examinations", methods=["GET"])
@role_required("admin")
def admin_examinations_page():
    examinations = get_all_examinations(limit=200)
    return render_template("admin/examinations.html", examinations=examinations)


@app.route("/admin/patients", methods=["GET"])
@role_required("admin")
def admin_patients_page():
    patients = get_all_patients(limit=200)
    return render_template("admin/patients.html", patients=patients)


# --------------------------------------------------
# PORTAL 2: HOSPITAL STAFF PORTAL
# --------------------------------------------------

@app.route("/staff/dashboard", methods=["GET"])
@role_required("staff", "admin")
def staff_dashboard():
    staff_user_id = session.get("user_id")
    recent_examinations = get_examinations_for_staff(staff_user_id, limit=20)
    doctors = get_active_doctors()
    patients = get_all_patients(limit=15)
    return render_template(
        "staff/dashboard.html",
        recent_examinations=recent_examinations,
        doctors=doctors,
        patients=patients,
    )


@app.route("/staff/search-patients", methods=["GET"])
@role_required("staff", "admin")
def staff_search_patients():
    query = request.args.get("q", "").strip()
    patients = search_patients(query)
    return jsonify({"success": True, "patients": patients})


@app.route("/staff/create-examination", methods=["POST"])
@role_required("staff", "admin")
def staff_create_examination():
    data = request.get_json(silent=True) or request.form
    patient_id = data.get("patient_id")
    doctor_id = data.get("doctor_id")
    notes = data.get("notes", "")

    if not patient_id or not doctor_id:
        return jsonify({"success": False, "error": "Patient and assigned Doctor must be selected."}), 400

    patient = get_patient(int(patient_id))
    doctor = get_doctor_by_id(int(doctor_id))

    if not patient:
        return jsonify({"success": False, "error": "Selected patient record not found."}), 404

    if not doctor or not doctor.get("is_active"):
        return jsonify({"success": False, "error": "Selected doctor account is not active."}), 400

    staff_user_id = session.get("user_id")
    examination_id = create_examination(
        patient_id=int(patient_id),
        doctor_id=int(doctor_id),
        staff_user_id=staff_user_id,
        notes=notes
    )

    if not examination_id:
        return jsonify({"success": False, "error": "Failed to create examination session."}), 500

    return jsonify({
        "success": True,
        "message": "Examination initialized.",
        "examination_id": examination_id,
        "next_url": f"/upload?patient_id={patient_id}&examination_id={examination_id}"
    }), 201


@app.route("/staff/patient/<int:patient_id>/summary", methods=["GET"])
@role_required("staff", "admin")
def staff_patient_summary(patient_id):
    patient = get_patient(patient_id)
    if not patient:
        return jsonify({"success": False, "error": "Patient not found."}), 404

    history = get_patient_examinations_history(patient_id)

    formatted_history = []
    for item in history:
        row = dict(item)
        if row.get("examined_at"):
            row["examined_at_str"] = row["examined_at"].strftime("%d %b %Y, %I:%M %p")
        else:
            row["examined_at_str"] = "N/A"
        if row.get("confidence_score") is not None:
            row["confidence_percent"] = round(float(row["confidence_score"]) * 100, 1)
        else:
            row["confidence_percent"] = None
        formatted_history.append(row)

    latest_exam = formatted_history[0] if formatted_history else None

    return jsonify({
        "success": True,
        "patient": patient,
        "has_history": len(formatted_history) > 0,
        "total_examinations": len(formatted_history),
        "latest_examination": latest_exam,
        "history": formatted_history
    })


# --------------------------------------------------
# PORTAL 3: DOCTOR PORTAL
# --------------------------------------------------

@app.route("/doctor/dashboard", methods=["GET"])
@role_required("doctor")
def doctor_dashboard():
    doctor_id = session.get("doctor_id")
    if not doctor_id:
        doc = get_doctor_by_user_id(session["user_id"])
        if doc:
            doctor_id = doc["doctor_id"]
            session["doctor_id"] = doctor_id
        else:
            return render_template("403.html", message="Doctor profile not found for this account."), 403

    examinations = get_examinations_for_doctor(doctor_id, limit=50)
    return render_template("doctor/dashboard.html", examinations=examinations)


@app.route("/doctor/search-patients", methods=["GET"])
@role_required("doctor")
def doctor_search_patients():
    query = request.args.get("q", "").strip()
    patients = search_patients(query)
    return jsonify({"success": True, "patients": patients})



def compute_examination_comparison(history):
    """
    Given a list of examinations sorted newest-first (history[0] is latest, history[1] is previous),
    computes objective previous-vs-current comparison metrics and statement without clinical bias.
    """
    if not history or len(history) < 2:
        return {
            "has_comparison": False,
            "message": "At least two examinations are required to perform a previous-versus-current comparison."
        }

    latest = history[0]
    previous = history[1]

    # Confidence formatting
    latest_conf = float(latest["confidence_score"] * 100) if latest.get("confidence_score") is not None else None
    prev_conf = float(previous["confidence_score"] * 100) if previous.get("confidence_score") is not None else None

    # Determine class change statement
    latest_disease = latest.get("detected_disease")
    prev_disease = previous.get("detected_disease")

    disease_summary = None
    if latest_disease and prev_disease:
        if latest_disease == prev_disease:
            disease_summary = "The predicted class is unchanged between the two examinations."
        else:
            disease_summary = f"The predicted class differs from the previous examination (previously '{prev_disease}', currently '{latest_disease}')."
    elif not latest_disease and not prev_disease:
        disease_summary = "Neither examination has AI prediction data available yet."
    elif not prev_disease:
        disease_summary = f"Previous examination had no prediction; current examination detected '{latest_disease}'."
    else:
        disease_summary = f"Previous examination detected '{prev_disease}'; current examination has no prediction recorded yet."

    # Determine confidence change statement
    conf_summary = None
    if latest_conf is not None and prev_conf is not None:
        diff = latest_conf - prev_conf
        if abs(diff) < 0.05:
            conf_summary = f"The AI confidence score is essentially unchanged ({prev_conf:.1f}% vs {latest_conf:.1f}%)."
        elif diff > 0:
            conf_summary = f"The AI confidence score increased from {prev_conf:.1f}% to {latest_conf:.1f}% (+{diff:.1f}%)."
        else:
            conf_summary = f"The AI confidence score decreased from {prev_conf:.1f}% to {latest_conf:.1f}% ({diff:.1f}%)."
    elif latest_conf is not None:
        conf_summary = f"Current examination AI confidence is {latest_conf:.1f}%; previous examination has no recorded confidence score."
    elif prev_conf is not None:
        conf_summary = f"Previous examination AI confidence was {prev_conf:.1f}%; current examination has no recorded confidence score."
    else:
        conf_summary = "Confidence scores are not available for either examination."

    safety_notice = (
        "AI results are intended to support screening and clinical review. "
        "Changes in predictions or confidence scores do not independently establish "
        "clinical improvement or deterioration. A qualified eye specialist should "
        "interpret the findings and determine appropriate next steps."
    )

    return {
        "has_comparison": True,
        "latest": latest,
        "previous": previous,
        "latest_conf": latest_conf,
        "prev_conf": prev_conf,
        "disease_summary": disease_summary,
        "conf_summary": conf_summary,
        "safety_notice": safety_notice,
    }


@app.route("/doctor/examination/<int:examination_id>", methods=["GET"])
@role_required("doctor")
def doctor_examination_detail(examination_id):
    doctor_id = session.get("doctor_id")
    if not doctor_id:
        doc = get_doctor_by_user_id(session["user_id"])
        if doc:
            doctor_id = doc["doctor_id"]
            session["doctor_id"] = doctor_id
        else:
            return abort(403)

    exam = get_examination_details(examination_id)
    if not exam:
        return abort(404)

    # Retrieve patient's full examination history
    patient_id = exam["patient_id"]
    history = get_patient_examinations_history(patient_id)

    # Compute comparison between the latest two examinations
    comparison = compute_examination_comparison(history)

    is_assigned = (exam.get("doctor_id") == doctor_id)

    return render_template(
        "doctor/examination_detail.html",
        exam=exam,
        history=history,
        comparison=comparison,
        is_assigned=is_assigned,
    )


@app.route("/doctor/patient/<int:patient_id>/history", methods=["GET"])
@role_required("doctor")
def doctor_patient_history(patient_id):
    doctor_id = session.get("doctor_id")
    if not doctor_id:
        doc = get_doctor_by_user_id(session["user_id"])
        if doc:
            doctor_id = doc["doctor_id"]
            session["doctor_id"] = doctor_id
        else:
            return abort(403)

    patient = get_patient(patient_id)
    if not patient:
        return abort(404)

    history = get_patient_examinations_history(patient_id)
    comparison = compute_examination_comparison(history)

    return render_template(
        "doctor/patient_history.html",
        patient=patient,
        history=history,
        comparison=comparison,
    )


# --------------------------------------------------
# PATIENT REGISTRATION (LEGACY & STAFF COMPATIBLE)
# --------------------------------------------------

@app.route("/register-patient", methods=["POST"])
def register_patient():
    data = request.get_json(silent=True) or request.form

    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "Please provide patient details."}), 400

    patient_name = str(data.get("patient_name", "")).strip()
    age_value = data.get("age")
    place = str(data.get("place", "")).strip()
    contact_number = str(data.get("contact_number", "")).strip()
    gender = str(data.get("gender", "")).strip() or None

    if not patient_name or len(patient_name) > 150:
        return jsonify({"success": False, "error": "Patient full name is required (max 150 chars)."}), 400

    if not place or len(place) > 150:
        return jsonify({"success": False, "error": "Place is required (max 150 chars)."}), 400

    try:
        age = int(age_value)
        if not 1 <= age <= 120:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Please enter a valid age between 1 and 120."}), 400

    if contact_number:
        # Validate contact number format: 7 to 20 chars, allowed digits, plus, hyphens, spaces, parens
        if not re.match(r"^\+?[0-9\s\-()]{7,20}$", contact_number):
            return jsonify({
                "success": False,
                "error": "Invalid contact number format. Please provide a valid phone number (e.g. +1-555-0199 or 9876543210)."
            }), 400

    try:
        patient_id = save_patient(
            patient_name=patient_name,
            age=age,
            place=place,
            contact_number=contact_number or None,
            gender=gender
        )
    except Exception as error:
        app.logger.exception("Patient registration failed: %s", error)
        return jsonify({"success": False, "error": "Unable to save patient details in database."}), 500

    if not patient_id:
        return jsonify({"success": False, "error": "Patient registration failed."}), 500

    return jsonify({
        "success": True,
        "message": "Patient registered successfully.",
        "patient_id": patient_id,
        "patient": {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "age": age,
            "place": place,
            "contact_number": contact_number,
            "gender": gender,
        },
        "next_page": f"/upload?patient_id={patient_id}",
    }), 201


# --------------------------------------------------
# FUNDUS IMAGE UPLOAD PAGE
# --------------------------------------------------

@app.route("/upload", methods=["GET"])
def upload_page():
    patient_id = request.args.get("patient_id", type=int)
    examination_id = request.args.get("examination_id", type=int)

    if not patient_id or patient_id < 1:
        return "Invalid patient ID. Please register or select the patient first.", 400

    try:
        patient = get_patient(patient_id)
    except Exception as error:
        app.logger.exception("Could not retrieve patient: %s", error)
        return "Unable to retrieve patient details.", 500

    if not patient:
        return "Patient not found. Please register or select the patient again.", 404

    active_doctors = get_active_doctors()

    return render_template(
        "upload.html",
        patient=patient,
        examination_id=examination_id,
        doctors=active_doctors
    )


# --------------------------------------------------
# RESULT PAGE
# --------------------------------------------------

@app.route("/result", methods=["GET"])
def result():
    return render_template("result.html")


# --------------------------------------------------
# PREDICT EYE DISEASE
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def handle_predict():
    # 1. Strictly restrict prediction execution: Doctors CANNOT run predictions
    if "role" in session and session.get("role") == "doctor":
        return jsonify({
            "success": False,
            "error": "Clinicians cannot run predictions directly. Predictions must be performed by intake hospital staff."
        }), 403

    # 2. Validate patient ID
    patient_id_value = request.form.get("patient_id", "").strip()
    examination_id_value = request.form.get("examination_id", "").strip()

    try:
        patient_id = int(patient_id_value)
        if patient_id < 1:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Valid patient ID is required."}), 400

    examination_id = None
    if examination_id_value:
        try:
            examination_id = int(examination_id_value)
        except ValueError:
            pass

    # 3. Retrieve patient
    patient = get_patient(patient_id)
    if not patient:
        return jsonify({"success": False, "error": "Patient not found."}), 404

    # 4. Validate uploaded image
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No fundus image was uploaded."}), 400

    file = request.files["image"]
    if not file or not file.filename or not file.filename.strip():
        return jsonify({"success": False, "error": "Please select a fundus image."}), 400

    original_filename = os.path.basename(file.filename)

    # 5. Verify image validity
    try:
        file.stream.seek(0)
        with Image.open(file.stream) as test_image:
            test_image.verify()
        file.stream.seek(0)
        with Image.open(file.stream) as opened_image:
            image = opened_image.convert("RGB")
    except Exception:
        return jsonify({"success": False, "error": "The uploaded file is not a valid image."}), 400

    # 5b. Validate whether image is a genuine retinal fundus photograph
    is_valid, validation_msg = validate_fundus_image(image)
    if not is_valid:
        return jsonify({
            "success": False,
            "error": f"Invalid Image: {validation_msg}"
        }), 400

    # 6. Load model
    try:
        model, device = get_model()
    except Exception as error:
        app.logger.exception("Model loading failed: %s", error)
        return jsonify({"success": False, "error": "The AI model could not be loaded."}), 500

    # 7. Predict disease
    try:
        (
            predicted_class,
            confidence,
            predicted_idx,
            input_tensor,
            rgb_image,
        ) = predict_with_details(image, model=model, device=device)
    except ValueError as val_err:
        return jsonify({"success": False, "error": f"Invalid Image: {str(val_err)}"}), 400
    except Exception as error:
        app.logger.exception("Prediction failed: %s", error)
        return jsonify({"success": False, "error": "Prediction failed. Please try another image."}), 500

    # 8. Generate Grad-CAM heatmap
    try:
        gradcam_filename = generate_gradcam(
            model=model,
            input_tensor=input_tensor,
            rgb_image=rgb_image,
            target_class_idx=predicted_idx,
            output_dir=DEFAULT_GENERATED_DIR,
        )
    except Exception as error:
        app.logger.exception("Grad-CAM generation failed: %s", error)
        return jsonify({"success": False, "error": "Could not generate Grad-CAM heatmap."}), 500

    # 9. Save uploaded image safely
    try:
        extension = os.path.splitext(original_filename)[1].lower()
        if extension not in (".jpg", ".jpeg", ".png", ".webp"):
            extension = ".jpg"

        saved_filename = f"{uuid.uuid4().hex}{extension}"
        saved_image_path = os.path.join(UPLOAD_DIR, saved_filename)
        image_format = {".png": "PNG", ".webp": "WEBP"}.get(extension, "JPEG")
        image.save(saved_image_path, format=image_format)
    except Exception as error:
        app.logger.exception("Image saving failed: %s", error)
        return jsonify({"success": False, "error": "Could not save uploaded image."}), 500

    # 10. Record in MySQL (eye_images & ai_predictions)
    try:
        prediction_id = save_prediction(
            image_name=original_filename,
            image_path=os.path.relpath(saved_image_path, project_root),
            detected_disease=predicted_class,
            confidence_score=confidence,
            model_version="EfficientNetB0",
            heatmap_url=gradcam_filename,
            patient_id=patient_id,
            examination_id=examination_id
        )
    except Exception as error:
        app.logger.exception("Saving prediction failed: %s", error)
        return jsonify({"success": False, "error": "Could not save prediction to database."}), 500

    if not prediction_id:
        return jsonify({"success": False, "error": "Prediction record could not be saved."}), 500

    # 11. Generate PDF Clinical Report
    report_url = None
    report_error = None
    try:
        gradcam_path = os.path.join(DEFAULT_GENERATED_DIR, os.path.basename(gradcam_filename))

        # Retrieve examination context if available
        doctor_name = "Clinical Review Pool"
        staff_username = session.get("username", "Hospital Staff")
        if examination_id:
            exam_data = get_examination_details(examination_id)
            if exam_data:
                doctor_name = exam_data.get("doctor_name", doctor_name)
                staff_username = exam_data.get("staff_username", staff_username)

        pdf_path = generate_pdf_report(
            prediction_id=prediction_id,
            image_name=original_filename,
            original_image_path=saved_image_path,
            predicted_disease=predicted_class,
            confidence=confidence,
            gradcam_image_path=gradcam_path,
            model_version="EfficientNetB0",
            patient_name=patient["patient_name"],
            patient_age=patient["age"],
            patient_place=patient["place"],
            patient_id=patient_id,
            examination_id=examination_id or "N/A",
            doctor_name=doctor_name,
            staff_username=staff_username
        )

        if pdf_path and os.path.isfile(pdf_path):
            relative_report_path = os.path.relpath(pdf_path, project_root)
            updated = update_report_path(prediction_id, relative_report_path)
            if updated:
                report_url = f"/download-report/{prediction_id}"
            else:
                report_error = "Could not link report in database."
        else:
            report_error = "PDF generator did not produce a file on disk."
    except Exception as error:
        report_error = f"PDF generation failed: {error}"
        app.logger.exception("PDF report generation failed: %s", error)

    return jsonify({
        "success": True,
        "patient": {
            "patient_id": patient_id,
            "patient_name": patient["patient_name"],
            "age": patient["age"],
            "place": patient["place"],
        },
        "prediction": predicted_class,
        "confidence": confidence,
        "gradcam_image": gradcam_filename,
        "prediction_id": prediction_id,
        "examination_id": examination_id,
        "report_url": report_url,
        "report_error": report_error,
    }), 200


# SERVE UPLOADED FUNDUS SCANS (PROTECTED)
# --------------------------------------------------

@app.route("/backend/uploads/<path:filename>", methods=["GET"])
@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_uploads(filename):
    # Ensure safe path traversal
    safe_path = os.path.realpath(os.path.join(UPLOAD_DIR, filename))
    if not safe_path.startswith(os.path.realpath(UPLOAD_DIR)):
        abort(403)

    return send_from_directory(UPLOAD_DIR, filename)


# --------------------------------------------------
# SERVE GRAD-CAM IMAGES (AUTHENTICATED / PROTECTED)
# --------------------------------------------------

@app.route("/generated/<path:filename>", methods=["GET"])
def serve_generated(filename):
    # Ensure safe path traversal
    safe_path = os.path.realpath(os.path.join(DEFAULT_GENERATED_DIR, filename))
    if not safe_path.startswith(os.path.realpath(DEFAULT_GENERATED_DIR)):
        abort(403)

    return send_from_directory(DEFAULT_GENERATED_DIR, filename)


# --------------------------------------------------
# DOWNLOAD PDF REPORT (STRICT AUTHORIZATION CHECKS)
# --------------------------------------------------

@app.route("/download-report/<int:prediction_id>", methods=["GET"])
def download_report(prediction_id):
    pred = get_prediction_by_id(prediction_id)
    if not pred or not pred.get("report_path"):
        return jsonify({"success": False, "error": "Report not found."}), 404

    # Strict Doctor Authorization Check: If caller is a logged in doctor, verify assignment
    if "role" in session and session.get("role") == "doctor":
        caller_doc_id = session.get("doctor_id")
        if pred.get("doctor_id") and pred["doctor_id"] != caller_doc_id:
            abort(403)

    report_path = pred["report_path"]
    reports_dir = os.path.realpath(REPORT_DIR)

    if os.path.isabs(report_path):
        resolved_path = os.path.realpath(report_path)
    else:
        resolved_path = os.path.realpath(os.path.join(project_root, report_path))

    # Path traversal validation
    try:
        if os.path.commonpath([reports_dir, resolved_path]) != reports_dir:
            abort(403)
    except ValueError:
        abort(403)

    if not os.path.isfile(resolved_path):
        return jsonify({"success": False, "error": "Report file is missing from disk."}), 404

    return send_file(
        resolved_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"VisionX_AI_Report_{prediction_id}.pdf",
    )


# --------------------------------------------------
# ERROR HANDLERS
# --------------------------------------------------

@app.errorhandler(403)
def forbidden(e):
    return render_template("403.html", message="Access Forbidden: You do not have permission to view this resource."), 403


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"success": False, "error": "Image is too large. Maximum size is 15 MB."}), 413


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )