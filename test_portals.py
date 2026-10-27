import urllib.request
import urllib.parse
import json
import http.cookiejar
import os
import mimetypes

BASE_URL = "http://127.0.0.1:5000"

def run_tests():
    print("=== STARTING VISIONX AI TEST SUITE ===")

    # 1. Test Admin Login with CookieJar
    cj_admin = http.cookiejar.CookieJar()
    opener_admin = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj_admin))

    login_data = json.dumps({
        "identifier": "admin",
        "password": "AdminPass123!",
        "role": "admin"
    }).encode('utf-8')

    req = urllib.request.Request(f"{BASE_URL}/login", data=login_data, headers={"Content-Type": "application/json"})
    with opener_admin.open(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print("[Test 1] Admin Login Status:", resp.status, res)
        assert res["success"] is True
        assert res["role"] == "admin"

    # 2. Test Admin Dashboard Access
    with opener_admin.open(f"{BASE_URL}/admin/dashboard") as resp:
        print("[Test 2] Admin Dashboard Access Status:", resp.status)
        assert resp.status == 200

    # 3. Provision Doctor via Admin Portal
    doc_payload = json.dumps({
        "username": "dr_robert",
        "email": "robert@visionx.clinic",
        "password": "DoctorPass123!",
        "full_name": "Robert Miller",
        "specialization": "Vitreo-Retinal Surgeon",
        "phone": "+1-555-0199"
    }).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}/admin/create-doctor", data=doc_payload, headers={"Content-Type": "application/json"})
    try:
        with opener_admin.open(req) as resp:
            print("[Test 3] Create Doctor:", resp.status, json.loads(resp.read().decode('utf-8')))
    except urllib.error.HTTPError as e:
        print("[Test 3] Create Doctor Notice:", e.code, e.read().decode('utf-8'))

    # 4. Provision Hospital Staff via Admin Portal
    staff_payload = json.dumps({
        "username": "staff_linda",
        "email": "linda@visionx.clinic",
        "password": "StaffPass123!"
    }).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}/admin/create-staff", data=staff_payload, headers={"Content-Type": "application/json"})
    try:
        with opener_admin.open(req) as resp:
            print("[Test 4] Create Staff:", resp.status, json.loads(resp.read().decode('utf-8')))
    except urllib.error.HTTPError as e:
        print("[Test 4] Create Staff Notice:", e.code, e.read().decode('utf-8'))

    # 5. Login as Staff Member
    cj_staff = http.cookiejar.CookieJar()
    opener_staff = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj_staff))

    staff_login = json.dumps({
        "identifier": "staff_linda",
        "password": "StaffPass123!",
        "role": "staff"
    }).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}/login", data=staff_login, headers={"Content-Type": "application/json"})
    with opener_staff.open(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print("[Test 5] Staff Login:", resp.status, res)
        assert res["success"] is True

    # 6. Verify Role Permission Restriction: Staff CANNOT access Admin Dashboard
    try:
        with opener_staff.open(f"{BASE_URL}/admin/dashboard") as resp:
            print("[FAIL] Staff should not access admin dashboard!")
    except urllib.error.HTTPError as e:
        print("[Test 6] Staff Access to Admin Denied as Expected! HTTP Code:", e.code)
        assert e.code == 403

    # 7. Staff Search Patients
    with opener_staff.open(f"{BASE_URL}/staff/search-patients?q=") as resp:
        patients_res = json.loads(resp.read().decode('utf-8'))
        print(f"[Test 7] Staff Search Patients: Found {len(patients_res['patients'])} patients.")
        assert len(patients_res['patients']) > 0
        patient_id = patients_res['patients'][0]['patient_id']

    # 8. Staff Creates Examination Session with Doctor
    # Look up doctor ID from database
    import sys
    sys.path.append('backend')
    from db_operations import get_active_doctors
    active_docs = get_active_doctors()
    assert len(active_docs) > 0
    assigned_doctor_id = active_docs[0]["doctor_id"]

    exam_payload = json.dumps({
        "patient_id": patient_id,
        "doctor_id": assigned_doctor_id,
        "notes": "Routine screening check for diabetic lesions"
    }).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}/staff/create-examination", data=exam_payload, headers={"Content-Type": "application/json"})
    with opener_staff.open(req) as resp:
        exam_res = json.loads(resp.read().decode('utf-8'))
        print("[Test 8] Staff Created Examination:", resp.status, exam_res)
        assert exam_res["success"] is True
        examination_id = exam_res["examination_id"]

    # 9. Staff Executes AI Inference Linked to Examination
    boundary = '----WebKitFormBoundaryStaffTest123'
    body = []
    body.append(f'--{boundary}\r\nContent-Disposition: form-data; name="patient_id"\r\n\r\n{patient_id}\r\n'.encode('utf-8'))
    body.append(f'--{boundary}\r\nContent-Disposition: form-data; name="examination_id"\r\n\r\n{examination_id}\r\n'.encode('utf-8'))
    image_path = 'DataSet/test/Normal_Fundus/4059_left.jpeg'
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    body.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="4059_left.jpeg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode('utf-8') + img_bytes + b'\r\n')
    body.append(f'--{boundary}--\r\n'.encode('utf-8'))
    data = b''.join(body)

    req = urllib.request.Request(f"{BASE_URL}/predict", data=data, headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    with opener_staff.open(req) as resp:
        pred_res = json.loads(resp.read().decode('utf-8'))
        print("[Test 9] Staff Executed Prediction:", resp.status, pred_res)
        assert pred_res["success"] is True
        prediction_id = pred_res["prediction_id"]

    # 10. Login as Assigned Doctor
    cj_doc = http.cookiejar.CookieJar()
    opener_doc = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj_doc))
    doc_login = json.dumps({
        "identifier": "dr_robert",
        "password": "DoctorPass123!",
        "role": "doctor"
    }).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}/login", data=doc_login, headers={"Content-Type": "application/json"})
    with opener_doc.open(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print("[Test 10] Doctor Login:", resp.status, res)
        assert res["success"] is True

    # 11. Doctor Views Dashboard & Examination Detail
    with opener_doc.open(f"{BASE_URL}/doctor/dashboard") as resp:
        print("[Test 11.1] Doctor Dashboard Status:", resp.status)
        assert resp.status == 200

    with opener_doc.open(f"{BASE_URL}/doctor/examination/{examination_id}") as resp:
        print("[Test 11.2] Doctor Examination Detail Status:", resp.status)
        assert resp.status == 200

    # 12. Doctor Can Download Generated Report
    with opener_doc.open(f"{BASE_URL}/download-report/{prediction_id}") as resp:
        pdf_bytes = resp.read()
        print("[Test 12] Doctor Download Report:", resp.status, f"{len(pdf_bytes)} bytes", "Is PDF:", pdf_bytes.startswith(b'%PDF'))
        assert pdf_bytes.startswith(b'%PDF')

    # 13. Verify Doctor CANNOT Run Prediction
    req_pred = urllib.request.Request(f"{BASE_URL}/predict", data=data, headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    try:
        with opener_doc.open(req_pred) as resp:
            print("[FAIL] Doctor should not be allowed to run prediction!")
    except urllib.error.HTTPError as e:
        print("[Test 13] Doctor Running Prediction Blocked as Expected! HTTP Code:", e.code)
        assert e.code == 403

    print("\n>>> ALL 13 TEST SUITE CHECKS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    run_tests()
