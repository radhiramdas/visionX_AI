/**
 * VisionX AI - Frontend Controller
 *
 * Handles:
 * 1. Legacy upload page support
 * 2. Patient information display
 * 3. Prediction and confidence display
 * 4. Original fundus image display
 * 5. Grad-CAM visualization
 * 6. PDF report download
 * 7. Navigation between screening pages
 */

// Human-friendly condition mapping for the seven trained retinal classes.
const CLASS_LABELS = {
    AMD: 'Age-Related Macular Degeneration (AMD)',
    Cataract: 'Cataract',
    DR: 'Diabetic Retinopathy (DR)',
    Glaucoma: 'Glaucoma',
    Hypertensive_Retinopathy: 'Hypertensive Retinopathy',
    Normal_Fundus: 'Normal Fundus (No Apparent Abnormality)',
    Pathological_Myopia: 'Pathological Myopia'
};


/**
 * Detect the current page and initialize its controller.
 */
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('upload-form')) {
        initUploadPage();
    }

    if (document.getElementById('result-container')) {
        initResultPage();
    }
});


/**
 * =========================================================================
 * 1. LEGACY UPLOAD PAGE CONTROLLER
 * =========================================================================
 *
 * Retained for compatibility with the original upload interface.
 * The newer upload.html page uses upload.js instead.
 */
function initUploadPage() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const previewFilename = document.getElementById('preview-filename');
    const changeFileBtn = document.getElementById('change-file-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const errorBanner = document.getElementById('error-banner');
    const errorMessage = document.getElementById('error-message');

    // Stop safely if the current HTML does not use the legacy upload UI.
    if (
        !dropzone ||
        !fileInput ||
        !previewContainer ||
        !previewImage ||
        !analyzeBtn ||
        !loadingOverlay ||
        !errorBanner ||
        !errorMessage
    ) {
        return;
    }

    let selectedFile = null;
    let selectedFileBase64 = null;

    function showError(message) {
        errorMessage.textContent = message;
        errorBanner.style.display = 'flex';
        loadingOverlay.style.display = 'none';
        analyzeBtn.disabled = selectedFile === null;
    }

    function clearError() {
        errorBanner.style.display = 'none';
        errorMessage.textContent = '';
    }

    function handleFile(file) {
        clearError();

        if (!file) {
            return;
        }

        const validTypes = [
            'image/jpeg',
            'image/jpg',
            'image/png'
        ];

        const filename = file.name.toLowerCase();

        const validExtension =
            filename.endsWith('.jpg') ||
            filename.endsWith('.jpeg') ||
            filename.endsWith('.png');

        if (!validTypes.includes(file.type) && !validExtension) {
            selectedFile = null;
            selectedFileBase64 = null;

            showError(
                'Please select a JPG, JPEG, or PNG retinal fundus image.'
            );

            return;
        }

        selectedFile = file;

        const reader = new FileReader();

        reader.onload = (event) => {
            selectedFileBase64 = event.target.result;
            previewImage.src = selectedFileBase64;

            if (previewFilename) {
                previewFilename.textContent =
                    `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            }

            dropzone.style.display = 'none';
            previewContainer.style.display = 'block';
            analyzeBtn.disabled = false;
        };

        reader.onerror = () => {
            showError('Unable to read the selected image. Please try again.');
        };

        reader.readAsDataURL(file);
    }

    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (event) => {
        if (event.target.files && event.target.files[0]) {
            handleFile(event.target.files[0]);
        }
    });

    if (changeFileBtn) {
        changeFileBtn.addEventListener('click', (event) => {
            event.preventDefault();

            selectedFile = null;
            selectedFileBase64 = null;
            fileInput.value = '';

            previewContainer.style.display = 'none';
            dropzone.style.display = 'block';
            analyzeBtn.disabled = true;

            clearError();
        });
    }

    ['dragenter', 'dragover'].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            event.stopPropagation();
            dropzone.classList.add('drag-active');
        });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            event.stopPropagation();
            dropzone.classList.remove('drag-active');
        });
    });

    dropzone.addEventListener('drop', (event) => {
        const files = event.dataTransfer.files;

        if (files && files[0]) {
            handleFile(files[0]);
        }
    });

    analyzeBtn.addEventListener('click', async (event) => {
        event.preventDefault();

        if (!selectedFile) {
            showError('Please select a retinal fundus image first.');
            return;
        }

        clearError();

        analyzeBtn.disabled = true;
        loadingOverlay.style.display = 'block';

        const formData = new FormData();
        formData.append('image', selectedFile);

        // Support the patient-registration workflow when a patient
        // has already been registered.
        try {
            const storedPatient = sessionStorage.getItem('visionx_patient');

            if (storedPatient) {
                const patient = JSON.parse(storedPatient);

                if (patient.patient_id != null) {
                    formData.append('patient_id', patient.patient_id);
                }
            }
        } catch (error) {
            console.error('Unable to read patient information:', error);
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                showError(
                    data.error ||
                    'Prediction failed. Please check the image and server.'
                );
                return;
            }

            sessionStorage.setItem(
                'visionx_result',
                JSON.stringify(data)
            );

            if (selectedFileBase64) {
                sessionStorage.setItem(
                    'visionx_original_image',
                    selectedFileBase64
                );
            }

            window.location.href = '/result';

        } catch (error) {
            console.error('Prediction request failed:', error);

            showError(
                'Unable to contact the backend. Check that Flask is running.'
            );
        }
    });
}


/**
 * =========================================================================
 * 2. RESULTS PAGE CONTROLLER
 * =========================================================================
 */
function initResultPage() {
    const rawResult = sessionStorage.getItem('visionx_result');
    const rawPatient = sessionStorage.getItem('visionx_patient');

    let originalImageSrc =
        sessionStorage.getItem('visionx_original_image');

    const urlParams = new URLSearchParams(window.location.search);

    let result = null;
    let patient = null;

    // Read the saved prediction.
    if (rawResult) {
        try {
            result = JSON.parse(rawResult);
        } catch (error) {
            console.error('Unable to parse screening result:', error);
        }
    }

    // Read the registered patient.
    if (rawPatient) {
        try {
            patient = JSON.parse(rawPatient);
        } catch (error) {
            console.error('Unable to parse patient information:', error);
        }
    }

    // Optional support for legacy query-string results.
    if (!result && (
        urlParams.has('prediction') ||
        urlParams.has('gradcam_image')
    )) {
        result = {
            success: true,
            prediction: urlParams.get('prediction') || 'Unknown',
            confidence: Number(urlParams.get('confidence')),
            gradcam_image: urlParams.get('gradcam_image') || ''
        };

        if (urlParams.has('original_image')) {
            originalImageSrc = urlParams.get('original_image');
        }
    }

    const noDataCard = document.getElementById('no-data-card');
    const resultCard = document.getElementById('result-card');

    // A valid result must exist before rendering the page.
    if (!result || result.success === false) {
        if (noDataCard) {
            noDataCard.style.display = 'block';
        }

        if (resultCard) {
            resultCard.style.display = 'none';
        }

        return;
    }

    /*
     * PATIENT INFORMATION
     *
     * Prefer the patient object returned by Flask.
     * If it is absent, use the registration data in sessionStorage.
     */
    const resultPatient = result.patient || patient || {};

    const patientFields = {
        'result-patient-id': resultPatient.patient_id,
        'result-patient-name': resultPatient.patient_name,
        'result-patient-age': resultPatient.age,
        'result-patient-place': resultPatient.place,
        'result-exam-id': result.examination_id ? `#${result.examination_id}` : '--',
        'result-exam-date': result.examined_at || new Date().toLocaleString([], { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
    };

    const historyBtn = document.getElementById('btn-patient-history');
    if (historyBtn && resultPatient.patient_id) {
        historyBtn.href = `/staff/dashboard?patient_id=${resultPatient.patient_id}`;
    }

    Object.entries(patientFields).forEach(([elementId, value]) => {
        const element = document.getElementById(elementId);

        if (!element) {
            return;
        }

        element.textContent =
            value !== undefined && value !== null && value !== ''
                ? String(value)
                : '--';
    });

    /*
     * PREDICTED DISEASE
     */
    const predictedClass =
        result.prediction || 'Unknown';

    const predictedClassElement =
        document.getElementById('predicted-class');

    const predictedFullnameElement =
        document.getElementById('predicted-fullname');

    if (predictedClassElement) {
        predictedClassElement.textContent = predictedClass;
    }

    if (predictedFullnameElement) {
        predictedFullnameElement.textContent =
            CLASS_LABELS[predictedClass] || predictedClass;
    }

    /*
     * CONFIDENCE SCORE
     */
    const confidenceElement =
        document.getElementById('confidence-value');

    const confidenceBar =
        document.getElementById('confidence-bar-fill');

    const parsedConfidence = Number(result.confidence);

    const validConfidence =
        result.confidence !== null &&
            result.confidence !== '' &&
            Number.isFinite(parsedConfidence)
            ? Math.min(Math.max(parsedConfidence, 0), 100)
            : null;

    if (confidenceElement) {
        confidenceElement.textContent =
            validConfidence !== null
                ? `${validConfidence.toFixed(1)}%`
                : 'Unavailable';
    }

    if (confidenceBar) {
        confidenceBar.style.width =
            validConfidence !== null
                ? `${validConfidence}%`
                : '0%';
    }

    /*
     * ORIGINAL FUNDUS IMAGE
     */
    const originalImageElement =
        document.getElementById('original-image');

    if (originalImageElement) {
        if (originalImageSrc) {
            originalImageElement.src = originalImageSrc;
        } else if (result.original_image_url) {
            originalImageElement.src = result.original_image_url;
        } else {
            originalImageElement.removeAttribute('src');
        }
    }

    /*
     * GRAD-CAM IMAGE
     *
     * Flask may return a URL or just the generated filename.
     */
    const gradcamElement =
        document.getElementById('gradcam-image');

    if (gradcamElement && result.gradcam_image) {
        let gradcamUrl = result.gradcam_image;

        if (
            !gradcamUrl.startsWith('/') &&
            !/^https?:\/\//i.test(gradcamUrl)
        ) {
            gradcamUrl =
                `/generated/${encodeURIComponent(gradcamUrl)}`;
        }

        gradcamElement.src = gradcamUrl;
    } else if (gradcamElement) {
        gradcamElement.removeAttribute('src');
    }

    /*
     * PDF REPORT DOWNLOAD
     *
     * The backend should return report_url, for example:
     * /download-report/12
     *
     * This button does not create a PDF itself.
     */
    const downloadButton =
        document.getElementById('download-report-btn');

    const reportMessage =
        document.getElementById('report-message');

    const reportUrl = result.report_url || (result.prediction_id ? `/download-report/${result.prediction_id}` : null);

    if (downloadButton) {
        if (result.report_url) {
            downloadButton.href = result.report_url;
            downloadButton.style.pointerEvents = 'auto';
            downloadButton.style.opacity = '1';
            downloadButton.removeAttribute('aria-disabled');
            if (reportMessage) {
                reportMessage.style.display = 'none';
            }
        } else if (result.report_error) {
            downloadButton.href = '#';
            downloadButton.style.pointerEvents = 'auto';
            downloadButton.style.opacity = '0.6';
            downloadButton.setAttribute('aria-disabled', 'true');

            if (reportMessage) {
                reportMessage.textContent = `Report generation issue: ${result.report_error}`;
                reportMessage.style.display = 'block';
                reportMessage.style.color = '#e53e3e';
            }
        } else {
            downloadButton.href = '#';
            downloadButton.style.pointerEvents = 'auto';
            downloadButton.style.opacity = '0.6';
            downloadButton.setAttribute('aria-disabled', 'true');

            if (reportMessage) {
                reportMessage.textContent =
                    'The PDF report is not available for this screening yet.';
                reportMessage.style.display = 'block';
            }
        }

        downloadButton.addEventListener('click', (event) => {
            if (!result.report_url) {
                event.preventDefault();

                if (reportMessage) {
                    reportMessage.textContent =
                        result.report_error
                            ? `Report error: ${result.report_error}`
                            : 'The PDF report is not available for this screening yet.';
                    reportMessage.style.display = 'block';
                }
            }
        });
    }

    /*
     * DISPLAY RESULTS
     */
    if (noDataCard) {
        noDataCard.style.display = 'none';
    }

    if (resultCard) {
        resultCard.style.display = 'block';
    }

    /*
     * ANALYZE ANOTHER IMAGE
     *
     * Return to patient registration and clear the previous session.
     */
    const analyzeAnotherButton =
        document.getElementById('analyze-another-btn');

    if (analyzeAnotherButton) {
        analyzeAnotherButton.addEventListener('click', () => {
            sessionStorage.removeItem('visionx_result');
            sessionStorage.removeItem('visionx_original_image');
            sessionStorage.removeItem('visionx_patient');

            window.location.href = '/';
        });
    }
}