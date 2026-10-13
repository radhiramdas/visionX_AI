import os
import sys
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image, UnidentifiedImageError

# Ensure backend directory is in sys.path for robust module resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Frontend directories (templates and static assets)
project_root = os.path.dirname(current_dir)
frontend_dir = os.path.join(project_root, "frontend")
template_dir = os.path.join(frontend_dir, "templates")
static_dir = os.path.join(frontend_dir, "static")

from model_loader import load_model, get_model
from prediction import predict_with_details
from gradcam import generate_gradcam, DEFAULT_GENERATED_DIR

# Initialize Flask application pointing to the frontend template and static paths
app = Flask(
    __name__,
    template_folder=template_dir,
    static_folder=static_dir
)

# Preload the model once at application startup
try:
    print("[Flask] Initializing model loading...")
    load_model()
    print("[Flask] Model successfully loaded at startup.")
except Exception as e:
    print(f"[Flask Warning] Could not preload model at startup: {e}")


@app.route("/", methods=["GET"])
def index():
    """
    Renders the main VisionX AI fundus screening upload page.
    """
    return render_template("index.html")


@app.route("/result", methods=["GET"])
def result():
    """
    Renders the screening results page displaying predictions and Grad-CAM explainability.
    """
    return render_template("result.html")


@app.route("/generated/<path:filename>", methods=["GET"])
def serve_generated(filename):
    """
    Serves generated Grad-CAM images from the backend/generated/ directory.
    """
    return send_from_directory(DEFAULT_GENERATED_DIR, filename)


@app.route("/predict", methods=["POST"])
def handle_predict():
    """
    Endpoint for retinal disease prediction and Grad-CAM generation.
    Accepts an uploaded image file under the 'image' form field.
    
    Returns JSON:
    {
        "success": true,
        "prediction": "DR",
        "confidence": 100.0,
        "gradcam_image": "gradcam_20261006_102500_a1b2c3d4.jpg"
    }
    """
    # 1. Validate that the 'image' key exists in multipart form files
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided in request. Please use form field 'image'."
        }), 400

    file = request.files["image"]

    # 2. Check for empty filename
    if not file or file.filename.strip() == "":
        return jsonify({
            "success": False,
            "error": "No file selected for upload."
        }), 400

    # 3. Attempt to open and validate the image using PIL
    try:
        image = Image.open(file.stream)
        # Verify image integrity
        image.verify()
        # Reset stream position after verify() as recommended by PIL
        file.stream.seek(0)
        image = Image.open(file.stream)
    except (UnidentifiedImageError, OSError, Exception) as img_err:
        return jsonify({
            "success": False,
            "error": f"Uploaded file is not a valid image: {str(img_err)}"
        }), 400

    # 4. Ensure model is loaded
    try:
        model, device = get_model()
    except Exception as model_err:
        return jsonify({
            "success": False,
            "error": f"Model loading failed: {str(model_err)}"
        }), 500

    # 5. Perform inference (obtaining prediction, aligned tensor, and visualization array)
    try:
        predicted_class, confidence, predicted_idx, input_tensor, rgb_image = predict_with_details(
            image, model=model, device=device
        )
    except Exception as pred_err:
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(pred_err)}"
        }), 500

    # 6. Generate Grad-CAM visualization for the predicted class
    try:
        gradcam_filename = generate_gradcam(
            model=model,
            input_tensor=input_tensor,
            rgb_image=rgb_image,
            target_class_idx=predicted_idx,
            output_dir=DEFAULT_GENERATED_DIR
        )
    except Exception as cam_err:
        return jsonify({
            "success": False,
            "error": f"Grad-CAM generation failed: {str(cam_err)}"
        }), 500

    # 7. Return successful JSON response
    return jsonify({
        "success": True,
        "prediction": predicted_class,
        "confidence": confidence,
        "gradcam_image": gradcam_filename
    }), 200


if __name__ == "__main__":
    # Run the Flask development server on localhost:5000
    app.run(host="127.0.0.1", port=5000, debug=False)
