/**
 * VisionX AI - Frontend Controller
 * Handles image selection, previews, async prediction, Grad-CAM rendering, and state management.
 */

// Human-friendly condition mapping for the 7 trained retinal classes
const CLASS_LABELS = {
    'AMD': 'Age-Related Macular Degeneration (AMD)',
    'Cataract': 'Cataract',
    'DR': 'Diabetic Retinopathy (DR)',
    'Glaucoma': 'Glaucoma',
    'Hypertensive_Retinopathy': 'Hypertensive Retinopathy',
    'Normal_Fundus': 'Normal Fundus (No Apparent Abnormality)',
    'Pathological_Myopia': 'Pathological Myopia'
};

document.addEventListener('DOMContentLoaded', () => {
    // Detect page context
    if (document.getElementById('upload-form')) {
        initUploadPage();
    } else if (document.getElementById('result-container')) {
        initResultPage();
    }
});

/**
 * =========================================================================
 * 1. Upload Page Controller (index.html)
 * =========================================================================
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

    let selectedFile = null;
    let selectedFileBase64 = null;

    // Helper: Show error banner
    function showError(msg) {
        errorMessage.textContent = msg;
        errorBanner.style.display = 'flex';
        loadingOverlay.style.display = 'none';
        analyzeBtn.disabled = selectedFile === null;
    }

    // Helper: Clear error banner
    function clearError() {
        errorBanner.style.display = 'none';
        errorMessage.textContent = '';
    }

    // Helper: Handle file selection
    function handleFile(file) {
        clearError();

        if (!file) return;

        // Validate image format (JPG/JPEG/PNG)
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        const fileName = file.name.toLowerCase();
        const hasValidExt = fileName.endsWith('.jpg') || fileName.endsWith('.jpeg') || fileName.endsWith('.png');

        if (!validTypes.includes(file.type) && !hasValidExt) {
            showError('Please select a valid retinal fundus image (.jpeg or .jpg format).');
            return;
        }

        selectedFile = file;

        // Read file for local preview and sessionStorage caching
        const reader = new FileReader();
        reader.onload = (e) => {
            selectedFileBase64 = e.target.result;
            previewImage.src = selectedFileBase64;
            previewFilename.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            
            dropzone.style.display = 'none';
            previewContainer.style.display = 'block';
            analyzeBtn.disabled = false;
        };
        reader.onerror = () => {
            showError('Unable to read selected file. Please try again.');
        };
        reader.readAsDataURL(file);
    }

    // Trigger file browser on click
    dropzone.addEventListener('click', () => fileInput.click());

    // File input change listener
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    // Change file button
    changeFileBtn.addEventListener('click', (e) => {
        e.preventDefault();
        selectedFile = null;
        selectedFileBase64 = null;
        fileInput.value = '';
        previewContainer.style.display = 'none';
        dropzone.style.display = 'block';
        analyzeBtn.disabled = true;
        clearError();
    });

    // Drag-and-drop events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('drag-active');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('drag-active');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files[0]) {
            handleFile(dt.files[0]);
        }
    });

    // Handle form submit / analyze button click
    analyzeBtn.addEventListener('click', async (e) => {
        e.preventDefault();

        if (!selectedFile) {
            showError('Please select a retinal fundus image before analyzing.');
            return;
        }

        clearError();
        analyzeBtn.disabled = true;
        loadingOverlay.style.display = 'block';

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                const errorText = data.error || 'Prediction request failed. Please check the image and server status.';
                showError(errorText);
                return;
            }

            // Store successful results and original image dataURL in sessionStorage
            sessionStorage.setItem('visionx_result', JSON.stringify(data));
            if (selectedFileBase64) {
                sessionStorage.setItem('visionx_original_image', selectedFileBase64);
            }

            // Redirect to results page
            window.location.href = '/result';

        } catch (err) {
            console.error('Inference error:', err);
            showError('Network error or backend unavailable. Please verify the Flask server is running.');
        }
    });
}

/**
 * =========================================================================
 * 2. Results Page Controller (result.html)
 * =========================================================================
 */
function initResultPage() {
    const rawResult = sessionStorage.getItem('visionx_result');
    let originalImageSrc = sessionStorage.getItem('visionx_original_image');

    const urlParams = new URLSearchParams(window.location.search);
    let result = null;

    if (rawResult) {
        try {
            result = JSON.parse(rawResult);
        } catch (e) {
            console.error('Error parsing session result:', e);
        }
    } else if (urlParams.has('prediction') || urlParams.has('gradcam_image')) {
        // Fallback support for query params (e.g. direct deep linking or automated test checks)
        result = {
            prediction: urlParams.get('prediction') || 'DR',
            confidence: parseFloat(urlParams.get('confidence')) || 100.0,
            gradcam_image: urlParams.get('gradcam_image') || ''
        };
        if (urlParams.has('original_image')) {
            originalImageSrc = urlParams.get('original_image');
        }
    }

    const noDataCard = document.getElementById('no-data-card');
    const resultCard = document.getElementById('result-card');

    if (!result) {
        if (noDataCard) noDataCard.style.display = 'block';
        if (resultCard) resultCard.style.display = 'none';
        return;
    }

    try {
        const predictedClassEl = document.getElementById('predicted-class');
        const predictedFullnameEl = document.getElementById('predicted-fullname');
        const confidenceValueEl = document.getElementById('confidence-value');
        const confidenceBarFillEl = document.getElementById('confidence-bar-fill');
        const originalImgEl = document.getElementById('original-image');
        const gradcamImgEl = document.getElementById('gradcam-image');

        // Populate Predicted Class
        const predClass = result.prediction || 'Unknown';
        if (predictedClassEl) predictedClassEl.textContent = predClass;
        if (predictedFullnameEl) {
            predictedFullnameEl.textContent = CLASS_LABELS[predClass] || predClass;
        }

        // Populate Confidence Percentage
        const confidenceVal = typeof result.confidence === 'number' ? result.confidence.toFixed(1) : result.confidence;
        if (confidenceValueEl) confidenceValueEl.textContent = `${confidenceVal}%`;
        if (confidenceBarFillEl) {
            confidenceBarFillEl.style.width = `${Math.min(Math.max(parseFloat(confidenceVal) || 0, 0), 100)}%`;
        }

        // Populate Original Image (or fallback to Grad-CAM preview if direct link)
        if (originalImgEl) {
            if (originalImageSrc) {
                originalImgEl.src = originalImageSrc;
            } else if (result.gradcam_image) {
                originalImgEl.src = `/generated/${encodeURIComponent(result.gradcam_image)}`;
            }
        }

        // Populate Grad-CAM Visualization
        if (gradcamImgEl && result.gradcam_image) {
            gradcamImgEl.src = `/generated/${encodeURIComponent(result.gradcam_image)}`;
        }

        if (noDataCard) noDataCard.style.display = 'none';
        if (resultCard) resultCard.style.display = 'block';

    } catch (e) {
        console.error('Error rendering result data:', e);
        if (noDataCard) noDataCard.style.display = 'block';
        if (resultCard) resultCard.style.display = 'none';
    }

    // Analyze Another Image button handler
    const backBtn = document.getElementById('analyze-another-btn');
    if (backBtn) {
        backBtn.addEventListener('click', () => {
            sessionStorage.removeItem('visionx_result');
            sessionStorage.removeItem('visionx_original_image');
            window.location.href = '/';
        });
    }
}
