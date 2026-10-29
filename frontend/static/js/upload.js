document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("upload-form");
    const imageInput = document.getElementById("fundus-image");
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");

    const analyzeButton = document.getElementById("analyze-btn");
    const errorBanner = document.getElementById("error-banner");
    const errorMessage = document.getElementById("error-message");
    const loadingMessage = document.getElementById("loading-message");

    // Get patient ID and examination ID from URL
    const params = new URLSearchParams(window.location.search);
    const patientId = Number(params.get("patient_id"));
    const examinationId = params.get("examination_id");

    // Retrieve registered patient details from session if available
    const storedPatient = sessionStorage.getItem("visionx_patient");
    let patient = null;
    try {
        patient = storedPatient ? JSON.parse(storedPatient) : null;
    } catch {
        patient = null;
    }

    // Validate patient information
    if (!Number.isInteger(patientId) || patientId < 1) {
        showError("Invalid patient ID. Please register or select a patient first.");
        if (analyzeButton) analyzeButton.disabled = true;
        return;
    }

    // Display patient details from sessionStorage if match
    if (patient && Number(patient.patient_id) === patientId) {
        const idElem = document.getElementById("patient-id");
        const nameElem = document.getElementById("patient-name");
        const ageElem = document.getElementById("patient-age");
        const placeElem = document.getElementById("patient-place");
        if (idElem) idElem.textContent = `#${patient.patient_id}`;
        if (nameElem) nameElem.textContent = patient.patient_name;
        if (ageElem) ageElem.textContent = patient.age;
        if (placeElem) placeElem.textContent = patient.place;
    }

    if (examinationId) {
        const examField = document.getElementById("exam-id-field");
        const examElem = document.getElementById("patient-exam-id");
        if (examField) examField.style.display = "block";
        if (examElem) examElem.textContent = `#${examinationId}`;
    }

    function showError(message) {
        if (errorBanner) errorBanner.style.display = "block";
        if (errorMessage) errorMessage.textContent = message;
    }

    function hideError() {
        if (errorBanner) errorBanner.style.display = "none";
        if (errorMessage) errorMessage.textContent = "";
    }

    // Preview selected image
    if (imageInput) {
        imageInput.addEventListener("change", () => {
            hideError();

            const file = imageInput.files[0];
            if (!file) {
                previewContainer.style.display = "none";
                imagePreview.removeAttribute("src");
                return;
            }

            const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
            if (!allowedTypes.includes(file.type)) {
                showError("Please select a valid JPG, PNG, or WebP retinal fundus image.");
                imageInput.value = "";
                previewContainer.style.display = "none";
                return;
            }

            if (file.size > 15 * 1024 * 1024) {
                showError("Image size must not exceed 15 MB.");
                imageInput.value = "";
                previewContainer.style.display = "none";
                return;
            }

            const reader = new FileReader();
            reader.onload = (event) => {
                imagePreview.src = event.target.result;
                previewContainer.style.display = "block";
            };
            reader.onerror = () => {
                showError("Could not preview this image. Please select another.");
            };
            reader.readAsDataURL(file);
        });
    }

    // Submit image, patient ID, and examination ID to Flask
    if (form) {
        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            hideError();

            const file = imageInput.files[0];
            if (!file) {
                showError("Please select a retinal fundus image.");
                return;
            }

            const formData = new FormData();
            formData.append("image", file);
            formData.append("patient_id", String(patientId));
            if (examinationId) {
                formData.append("examination_id", String(examinationId));
            }

            analyzeButton.disabled = true;
            analyzeButton.textContent = "Processing Inference...";
            if (loadingMessage) loadingMessage.style.display = "block";

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();

                if (!response.ok || !data.success) {
                    throw new Error(data.error || "Image analysis failed.");
                }

                // Save results for the results page
                sessionStorage.setItem("visionx_result", JSON.stringify(data));

                if (imagePreview.src && imagePreview.src.startsWith("data:")) {
                    sessionStorage.setItem("visionx_original_image", imagePreview.src);
                }

                // Navigate to the result page
                window.location.href = "/result";

            } catch (error) {
                showError(error.message || "An unexpected error occurred during prediction.");
                analyzeButton.disabled = false;
                analyzeButton.textContent = "Execute AI Retinal Screening";
                if (loadingMessage) loadingMessage.style.display = "none";
            }
        });
    }
});