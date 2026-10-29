import urllib.request
import urllib.parse
import urllib.error
import json
import http.cookiejar
import os
import sys

BASE_URL = "http://127.0.0.1:5000"

def create_opener():
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    return opener

def post_json(opener, url, data):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    with opener.open(req) as resp:
        return resp.status, json.loads(resp.read().decode('utf-8'))

def get_json(opener, url):
    req = urllib.request.Request(url)
    with opener.open(req) as resp:
        return resp.status, json.loads(resp.read().decode('utf-8'))

def get_bytes(opener, url):
    req = urllib.request.Request(url)
    with opener.open(req) as resp:
        return resp.status, resp.read()

def get_page(opener, url):
    req = urllib.request.Request(url)
    with opener.open(req) as resp:
        return resp.status, resp.read().decode('utf-8')

def run_all_tests():
    print("=" * 60)
    print("VISIONX AI - COMPREHENSIVE WORKFLOW & VERIFICATION SUITE")
    print("=" * 60)

    # ---------------------------------------------------------
    # TEST A: LOGIN & ROLE ACCESS CONTROL
    # ---------------------------------------------------------
    print("\n--- TEST A: LOGIN & ROLE ACCESS CONTROL ---")
    opener_staff = create_opener()
    opener_doc = create_opener()
    opener_admin = create_opener()

    # 1. Staff Login
    status, res = post_json(opener_staff, f"{BASE_URL}/login", {
        "identifier": "staff_linda",
        "password": "StaffPass123!",
        "role": "staff"
    })
    assert status == 200 and res["success"] is True and res["role"] == "staff"
    print(" [A.1] Staff login successful:", res["message"])

    # 2. Doctor Login
    status, res = post_json(opener_doc, f"{BASE_URL}/login", {
        "identifier": "dr_robert",
        "password": "DoctorPass123!",
        "role": "doctor"
    })
    assert status == 200 and res["success"] is True and res["role"] == "doctor"
    print(" [A.2] Doctor login successful:", res["message"])

    # 3. Admin Login
    status, res = post_json(opener_admin, f"{BASE_URL}/login", {
        "identifier": "admin",
        "password": "AdminPass123!",
        "role": "admin"
    })
    assert status == 200 and res["success"] is True and res["role"] == "admin"
    print(" [A.3] Admin login successful:", res["message"])

    # 4. Invalid credentials rejected
    opener_bad = create_opener()
    try:
        post_json(opener_bad, f"{BASE_URL}/login", {
            "identifier": "staff_linda",
            "password": "WrongPassword!",
            "role": "staff"
        })
        assert False, "Should have rejected wrong password"
    except urllib.error.HTTPError as e:
        assert e.code == 401
        print(" [A.4] Invalid credentials correctly rejected (HTTP 401).")

    # 5. Incorrect role selection rejected
    try:
        post_json(opener_bad, f"{BASE_URL}/login", {
            "identifier": "dr_robert",
            "password": "DoctorPass123!",
            "role": "staff"
        })
        assert False, "Should have rejected mismatched role"
    except urllib.error.HTTPError as e:
        assert e.code == 403
        print(" [A.5] Incorrect role selection correctly rejected (HTTP 403).")

    # 6. Unauthorized dashboard access blocked
    try:
        get_page(opener_staff, f"{BASE_URL}/doctor/dashboard")
        assert False, "Staff must not access doctor dashboard"
    except urllib.error.HTTPError as e:
        assert e.code == 403
        print(" [A.6] Staff entering doctor dashboard correctly blocked (HTTP 403).")

    try:
        get_page(opener_doc, f"{BASE_URL}/admin/dashboard")
        assert False, "Doctor must not access admin dashboard"
    except urllib.error.HTTPError as e:
        assert e.code == 403
        print(" [A.7] Doctor entering admin dashboard correctly blocked (HTTP 403).")

    # ---------------------------------------------------------
    # TEST B: NEW PATIENT REGISTRATION
    # ---------------------------------------------------------
    print("\n--- TEST B: NEW PATIENT REGISTRATION ---")
    test_patient_name = "Eleanor Vance"
    status, reg_res = post_json(opener_staff, f"{BASE_URL}/register-patient", {
        "patient_name": test_patient_name,
        "age": 54,
        "gender": "Female",
        "contact_number": "+1-555-0144",
        "place": "Metropolitan Eye Institute, NY"
    })
    assert status == 201 and reg_res["success"] is True
    new_patient_id = reg_res["patient_id"]
    assert new_patient_id > 0
    print(f" [B.1] Patient registered successfully with Unique Patient ID: #{new_patient_id}")
    print(f"       Details: {reg_res['patient']}")

    # Validate phone format rejection
    try:
        post_json(opener_staff, f"{BASE_URL}/register-patient", {
            "patient_name": "Invalid Phone Test",
            "age": 40,
            "gender": "Male",
            "contact_number": "invalid-phone-abc",
            "place": "Test Hospital"
        })
        assert False, "Should reject invalid phone format"
    except urllib.error.HTTPError as e:
        assert e.code == 400
        print(" [B.2] Invalid phone format correctly rejected (HTTP 400).")

    # ---------------------------------------------------------
    # TEST C: EXISTING PATIENT SEARCH & SUMMARY
    # ---------------------------------------------------------
    print("\n--- TEST C: EXISTING PATIENT SEARCH & SUMMARY ---")
    # 1. Search by Patient ID
    status, s_res = get_json(opener_staff, f"{BASE_URL}/staff/search-patients?q={new_patient_id}")
    assert status == 200 and len(s_res["patients"]) > 0
    found_patient = s_res["patients"][0]
    assert found_patient["patient_id"] == new_patient_id
    print(f" [C.1] Found patient by ID #{new_patient_id}: {found_patient['patient_name']}, Contact: {found_patient.get('contact_number')}")

    # 2. Search by Name
    status, s_res2 = get_json(opener_staff, f"{BASE_URL}/staff/search-patients?q=Eleanor")
    assert status == 200 and len(s_res2["patients"]) > 0
    print(f" [C.2] Found {len(s_res2['patients'])} patient(s) matching query 'Eleanor'.")

    # 3. Retrieve Summary for patient with 0 exams
    status, sum_res = get_json(opener_staff, f"{BASE_URL}/staff/patient/{new_patient_id}/summary")
    assert status == 200 and sum_res["has_history"] is False
    print(" [C.3] New patient summary correctly reports 0 previous examinations (has_history: False).")

    # ---------------------------------------------------------
    # TEST D: FIRST SCREENING FOR PATIENT
    # ---------------------------------------------------------
    print("\n--- TEST D: FIRST SCREENING (EXAMINATION 1) ---")
    # 1. Create Examination Session
    status, ex_res = post_json(opener_staff, f"{BASE_URL}/staff/create-examination", {
        "patient_id": new_patient_id,
        "doctor_id": 1,
        "notes": "Initial baseline retina checkup"
    })
    assert status == 201 and ex_res["success"] is True
    exam_1_id = ex_res["examination_id"]
    print(f" [D.1] Examination 1 created successfully: Exam #{exam_1_id} for Patient #{new_patient_id}")

    # 2. Execute AI Retinal Screening
    boundary = '----WebKitFormBoundaryStaffInference'
    body = []
    body.append(f'--{boundary}\r\nContent-Disposition: form-data; name="patient_id"\r\n\r\n{new_patient_id}\r\n'.encode('utf-8'))
    body.append(f'--{boundary}\r\nContent-Disposition: form-data; name="examination_id"\r\n\r\n{exam_1_id}\r\n'.encode('utf-8'))
    image_path = 'DataSet/test/Normal_Fundus/4059_left.jpeg'
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    body.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="4059_left.jpeg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode('utf-8') + img_bytes + b'\r\n')
    body.append(f'--{boundary}--\r\n'.encode('utf-8'))
    data = b''.join(body)

    req = urllib.request.Request(f"{BASE_URL}/predict", data=data, headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    with opener_staff.open(req) as resp:
        pred_res = json.loads(resp.read().decode('utf-8'))
        assert resp.status == 200 and pred_res["success"] is True
        pred_1_id = pred_res["prediction_id"]
        conf_val = pred_res["confidence"]
        if conf_val <= 1.0: conf_val *= 100.0
        print(f" [D.2] AI Inference completed: Disease={pred_res['prediction']}, Confidence={round(conf_val, 1)}%, PredID={pred_1_id}")
        assert pred_res["gradcam_image"] is not None
        print(f"       Grad-CAM Heatmap generated: {pred_res['gradcam_image']}")
        assert pred_res["report_url"] is not None
        print(f"       PDF Clinical Report generated: {pred_res['report_url']}")

    # ---------------------------------------------------------
    # TEST E: RETURNING PATIENT SUMMARY & SECOND SCREENING
    # ---------------------------------------------------------
    print("\n--- TEST E: RETURNING PATIENT SUMMARY & SECOND SCREENING ---")
    # Check patient summary now reports previous examination
    status, sum_res = get_json(opener_staff, f"{BASE_URL}/staff/patient/{new_patient_id}/summary")
    assert status == 200 and sum_res["has_history"] is True
    assert sum_res["total_examinations"] == 1
    assert sum_res["latest_examination"]["examination_id"] == exam_1_id
    print(f" [E.1] Returning patient summary accurately displays Previous Examination #{exam_1_id} with diagnosis '{sum_res['latest_examination']['detected_disease']}'")

    # Initiate SECOND screening (Visit 2 for returning patient)
    status, ex_res2 = post_json(opener_staff, f"{BASE_URL}/staff/create-examination", {
        "patient_id": new_patient_id,
        "doctor_id": 1,
        "notes": "Follow-up examination after 1 month"
    })
    assert status == 201 and ex_res2["success"] is True
    exam_2_id = ex_res2["examination_id"]
    assert exam_2_id != exam_1_id
    print(f" [E.2] Second Examination created: Exam #{exam_2_id} (Reusing Patient ID #{new_patient_id})")

    # Run AI inference on second visit
    body2 = []
    body2.append(f'--{boundary}\r\nContent-Disposition: form-data; name="patient_id"\r\n\r\n{new_patient_id}\r\n'.encode('utf-8'))
    body2.append(f'--{boundary}\r\nContent-Disposition: form-data; name="examination_id"\r\n\r\n{exam_2_id}\r\n'.encode('utf-8'))
    image_path2 = 'DataSet/test/Cataract/Cataract0_left_0_5312.jpeg'
    with open(image_path2, 'rb') as f:
        img_bytes2 = f.read()
    body2.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="Cataract0_left_0_5312.jpeg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode('utf-8') + img_bytes2 + b'\r\n')
    body2.append(f'--{boundary}--\r\n'.encode('utf-8'))
    data2 = b''.join(body2)

    req2 = urllib.request.Request(f"{BASE_URL}/predict", data=data2, headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    with opener_staff.open(req2) as resp:
        pred_res2 = json.loads(resp.read().decode('utf-8'))
        assert resp.status == 200 and pred_res2["success"] is True
        pred_2_id = pred_res2["prediction_id"]
        conf_val2 = pred_res2["confidence"]
        if conf_val2 <= 1.0: conf_val2 *= 100.0
        print(f" [E.3] Second Inference completed: Disease={pred_res2['prediction']}, Confidence={round(conf_val2, 1)}%, PredID={pred_2_id}")

    # Check updated summary: newest examination must appear first
    status, sum_res2 = get_json(opener_staff, f"{BASE_URL}/staff/patient/{new_patient_id}/summary")
    assert status == 200 and sum_res2["total_examinations"] == 2
    assert sum_res2["latest_examination"]["examination_id"] == exam_2_id
    assert sum_res2["history"][0]["examination_id"] == exam_2_id
    assert sum_res2["history"][1]["examination_id"] == exam_1_id
    print(f" [E.4] Examination history ordered newest-first: [Latest: #{exam_2_id}, Previous: #{exam_1_id}]. Older records intact!")

    # ---------------------------------------------------------
    # TEST F: DOCTOR WORKFLOW & LONGITUDINAL COMPARISON
    # ---------------------------------------------------------
    print("\n--- TEST F: DOCTOR WORKFLOW & COMPARISON ---")
    # 1. Doctor Dashboard loads
    status, html_doc_dash = get_page(opener_doc, f"{BASE_URL}/doctor/dashboard")
    assert status == 200 and "Assigned Clinical Examinations" in html_doc_dash
    print(" [F.1] Doctor Dashboard loaded successfully.")

    # 2. Doctor views Examination 2 Detail with Comparison against Examination 1
    status, html_exam_detail = get_page(opener_doc, f"{BASE_URL}/doctor/examination/{exam_2_id}")
    assert status == 200
    assert "Previous-versus-Current Examination Comparison" in html_exam_detail
    assert "Clinical Safety Notice" in html_exam_detail or "clinical review" in html_exam_detail
    print(" [F.2] Doctor Examination Detail loaded with longitudinal comparison and clinical safety notice.")

    # 3. Doctor views full patient history
    status, html_history = get_page(opener_doc, f"{BASE_URL}/doctor/patient/{new_patient_id}/history")
    assert status == 200
    assert "Patient Examination History" in html_history
    print(" [F.3] Doctor Patient History page loaded successfully.")

    # 4. Doctor downloads PDF reports for both examinations
    status, pdf1 = get_bytes(opener_doc, f"{BASE_URL}/download-report/{pred_1_id}")
    assert status == 200 and pdf1.startswith(b'%PDF')
    status, pdf2 = get_bytes(opener_doc, f"{BASE_URL}/download-report/{pred_2_id}")
    assert status == 200 and pdf2.startswith(b'%PDF')
    print(f" [F.4] Doctor downloaded both PDF reports (Pred #{pred_1_id}: {len(pdf1)} bytes & #{pred_2_id}: {len(pdf2)} bytes) successfully.")

    # ---------------------------------------------------------
    # TEST G: REGRESSION CHECKS
    # ---------------------------------------------------------
    print("\n--- TEST G: REGRESSION CHECKS ---")
    # Check that doctor cannot directly run predictions
    try:
        req_doc_pred = urllib.request.Request(f"{BASE_URL}/predict", data=data, headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
        with opener_doc.open(req_doc_pred) as resp:
            assert False, "Doctor must not run predictions"
    except urllib.error.HTTPError as e:
        assert e.code == 403
        print(" [G.1] Doctor direct prediction execution prohibited as expected (HTTP 403).")

    # Check admin dashboard
    status, html_admin = get_page(opener_admin, f"{BASE_URL}/admin/dashboard")
    assert status == 200
    print(" [G.2] Admin Dashboard operational.")

    # Check upload page displays patient and examination ID
    status, html_upload = get_page(opener_staff, f"{BASE_URL}/upload?patient_id={new_patient_id}&examination_id={exam_2_id}")
    assert status == 200 and str(new_patient_id) in html_upload and str(exam_2_id) in html_upload
    print(f" [G.3] Upload page correctly renders Patient ID #{new_patient_id} and Examination ID #{exam_2_id}.")

    print("\n" + "=" * 60)
    print(">>> ALL TEST SUITES (A through G) PASSED WITH ZERO ERRORS! <<<")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
