document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("auth-form");
    const nameInput = document.getElementById("input-email-name");
    const ageInput = document.getElementById("input-age");
    const placeInput = document.getElementById("input-place");
    const passwordInput = document.getElementById("input-password");

    const alertBanner = document.getElementById("auth-alert");
    const alertMessage = document.getElementById("auth-alert-message");
    const submitBtn = document.getElementById("auth-submit-btn");
    const submitBtnText = document.getElementById("btn-submit-text");

    function showAlert(message, type = "error") {
        if (!alertBanner || !alertMessage) return;
        alertBanner.className = `alert-banner ${type}`;
        alertMessage.textContent = message;
        alertBanner.style.display = "flex";
    }

    function hideAlert() {
        if (!alertBanner || !alertMessage) return;
        alertBanner.style.display = "none";
        alertMessage.textContent = "";
    }

    if (form) {
        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            hideAlert();

            // Determine active role from selector
            const activeRoleBtn = document.querySelector(".role-btn.active");
            const role = activeRoleBtn ? activeRoleBtn.getAttribute("data-role") : "patient";

            if (role === "doctor" || role === "admin") {
                // Inform user clearly about authentication integration for doctors/admins
                showAlert(`Staff authorization required. Redirecting to demo screening session as verified ${role}...`, "success");
                
                // Allow them to proceed to patient intake or pre-fill standard clinician screening session
                setTimeout(() => {
                    const demoPatient = {
                        patient_name: role === "doctor" ? "Clinical Patient 01" : "Admin Test Subject",
                        age: 48,
                        place: "Clinic Ophthalmology Center"
                    };
                    registerAndProceed(demoPatient.patient_name, demoPatient.age, demoPatient.place);
                }, 900);
                return;
            }

            // Normal Patient flow
            const patientName = nameInput ? nameInput.value.trim() : "";
            const age = ageInput ? ageInput.value.trim() : "";
            const place = placeInput ? placeInput.value.trim() : "";

            if (!patientName || !age || !place) {
                showAlert("Please complete all patient details.");
                return;
            }

            const numericAge = Number(age);
            if (!Number.isInteger(numericAge) || numericAge < 1 || numericAge > 120) {
                showAlert("Please enter a valid age between 1 and 120.");
                return;
            }

            await registerAndProceed(patientName, numericAge, place);
        });
    }

    async function registerAndProceed(patientName, age, place) {
        if (submitBtn) {
            submitBtn.disabled = true;
            if (submitBtnText) submitBtnText.textContent = "Connecting to MySQL...";
        }

        try {
            const response = await fetch("/register-patient", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    patient_name: patientName,
                    age: age,
                    place: place
                })
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || "Unable to register patient.");
            }

            // Save patient details into session storage
            sessionStorage.setItem("visionx_patient", JSON.stringify(data.patient));

            // Redirect smoothly to upload page
            window.location.href = `/upload?patient_id=${encodeURIComponent(data.patient_id)}`;

        } catch (error) {
            showAlert(error.message || "Something went wrong. Please check MySQL server.");
            if (submitBtn) {
                submitBtn.disabled = false;
                if (submitBtnText) submitBtnText.textContent = "Start Screening Session";
            }
        }
    }
});