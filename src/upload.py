# -*- coding: utf-8 -*-

import os
import traceback
import logging
from flask import Flask, request, render_template, send_from_directory, jsonify, url_for
from datetime import datetime # <--- Ensure datetime is imported
from werkzeug.utils import secure_filename

# Assuming your infer_by_web function is in main.py within the same src directory
try:
    # This function should return (prediction_string, probability_float)
    from main import infer_by_web
except ImportError as e:
    # Configure logging early in case of import issues
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.error(f"CRITICAL ERROR: Failed to import 'infer_by_web' from main.py: {e}")
    logging.error("Ensure main.py and its dependencies (Model.py, etc.) are in the 'src' directory and functional.")
    # Set infer_by_web to None so the application can still start (maybe)
    # but prediction attempts will fail gracefully.
    infer_by_web = None

__author__ = 'Sushant (Updated for Project Demo)'

# --- Logging Configuration ---
# Configure logging (this might be redundant if done above, but ensures it's set)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

app = Flask(__name__)

# --- Configuration ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__)) # src directory
STATIC_FOLDER = os.path.join(APP_ROOT, 'static')
# Relative paths within the static folder
UPLOAD_FOLDER_REL = 'uploads'
EXAMPLE_FOLDER_REL = 'examples'
# Absolute paths for backend use
UPLOAD_TARGET_ABS = os.path.join(STATIC_FOLDER, UPLOAD_FOLDER_REL)
EXAMPLE_FOLDER_ABS = os.path.join(STATIC_FOLDER, EXAMPLE_FOLDER_REL)

# Create target directories if they don't exist
try:
    os.makedirs(UPLOAD_TARGET_ABS, exist_ok=True)
    os.makedirs(EXAMPLE_FOLDER_ABS, exist_ok=True)
except OSError as e:
     logging.error(f"Error creating directories: {e}")
     # Consider if the app should exit if directories can't be made

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
# --- End Configuration ---


# --- Main Page Routes ---

@app.route("/")
def index():
    """Serves the Landing/Home page."""
    logging.info("Request received for / (Landing Page)")
    current_year = datetime.now().year
    return render_template("index.html", current_year=current_year)

@app.route("/recognize")
def recognizer():
    """Serves the main OCR tool page."""
    logging.info("Request received for /recognize (OCR Tool Page)")
    current_year = datetime.now().year
    return render_template("recognizer.html", current_year=current_year)

# --- Static Page Routes ---

@app.route("/about")
def about():
    """Serves the About page."""
    logging.info("Request received for /about")
    current_year = datetime.now().year
    return render_template("about.html", current_year=current_year)

@app.route("/technology")
def technology():
    """Serves the Technology page."""
    logging.info("Request received for /technology")
    current_year = datetime.now().year
    return render_template("technology.html", current_year=current_year)

@app.route("/examples")
def examples():
    """Serves the Examples page, processing images on the backend."""
    logging.info("Request received for /examples")
    current_year = datetime.now().year
    examples_data = []
    error_message = None

    # Check if prediction service is available *before* trying to process
    if infer_by_web is None:
        logging.error("Examples page cannot be fully rendered: Prediction service unavailable.")
        error_message = "Prediction service is currently unavailable. Cannot process examples."
        return render_template("examples.html", examples_data=[], error=error_message, current_year=current_year)

    try:
        if not os.path.exists(EXAMPLE_FOLDER_ABS):
            logging.warning(f"Example image directory not found: {EXAMPLE_FOLDER_ABS}")
            error_message = "Example directory not found."
        else:
            filenames = sorted([f for f in os.listdir(EXAMPLE_FOLDER_ABS)
                               if os.path.isfile(os.path.join(EXAMPLE_FOLDER_ABS, f)) and
                               os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS])

            if not filenames:
                logging.warning(f"No valid example images found in {EXAMPLE_FOLDER_ABS}")
                error_message = "No example images found."
            else:
                logging.info(f"Processing {len(filenames)} example images for display.")
                for filename in filenames:
                    file_path = os.path.join(EXAMPLE_FOLDER_ABS, filename)
                    logging.debug(f"Processing example: {file_path}")
                    try:
                        # Perform prediction for the example image
                        pred, prob = infer_by_web(file_path)
                        examples_data.append({
                            "filename": filename,
                            "url": url_for('static', filename=f"{EXAMPLE_FOLDER_REL}/{filename}"),
                            "prediction": pred,
                            "probability": prob if not isinstance(prob, str) else 0.0 # Handle potential errors
                        })
                    except Exception as infer_err:
                         # Log error for specific image but continue with others
                         logging.error(f"Error predicting example '{filename}': {infer_err}")
                         examples_data.append({
                            "filename": filename,
                            "url": url_for('static', filename=f"{EXAMPLE_FOLDER_REL}/{filename}"),
                            "prediction": f"Error: Processing failed",
                            "probability": 0.0
                         })

    except Exception as e:
        logging.error(f"Error loading examples page: {e}")
        logging.error(traceback.format_exc())
        error_message = "An error occurred while loading examples."

    # Pass year to the template context
    return render_template("examples.html", examples_data=examples_data, error=error_message, current_year=current_year)

# --- API Endpoints ---

@app.route("/predict", methods=["POST"])
def predict():
    """Handles file upload, prediction, and returns JSON result."""
    logging.info("Received POST request to /predict")

    # Check if prediction service is available
    if infer_by_web is None:
        logging.error("Prediction failed: Prediction service unavailable.")
        return jsonify({"error": "Prediction service is currently unavailable."}), 503 # Service Unavailable

    # Check if the post request has the file part
    if 'file' not in request.files:
        logging.error("No 'file' part in the request.")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files.get('file')

    if not file or not file.filename:
        logging.error("No selected file or empty filename.")
        return jsonify({"error": "No selected file provided"}), 400

    try:
        # Use secure_filename for safety and normalize filename
        filename = secure_filename(file.filename)
        logging.info(f"Received file: {filename}")

        # Validate file extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            logging.error(f"File type '{ext}' not supported for {filename}.")
            return jsonify({"error": f"File type not supported. Please use {', '.join(ALLOWED_EXTENSIONS)}."}), 400

        # Create a unique filename using timestamp to avoid overwrites
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Limit filename length if necessary before adding timestamp
        base, ext = os.path.splitext(filename)
        base = base[:50] # Example: limit base filename part
        unique_filename = f"{timestamp}_{base}{ext}"

        destination = os.path.join(UPLOAD_TARGET_ABS, unique_filename)
        logging.info(f"Saving file to: {destination}")
        file.save(destination)

        # Perform prediction
        logging.info(f"Calling prediction function for: {destination}")
        result, probability = infer_by_web(destination) # Call function from main.py
        logging.info(f"Prediction result: '{result}', Probability: {probability:.4f}")

        # Check if prediction itself returned an error message
        if isinstance(result, str) and result.startswith("Error:"):
             logging.error(f"Prediction function returned error: {result}")
             # Return a more generic server error to the client unless it's a specific user issue
             if "Image loading failed" in result:
                 return jsonify({"error": "Server could not load the image for processing."}), 500
             elif "Model not loaded" in result:
                  return jsonify({"error": "Prediction model is unavailable."}), 503
             else:
                  return jsonify({"error": "An error occurred during prediction."}), 500

        # Return JSON response on success
        return jsonify({
            "prediction": result,
            "probability": float(probability) # Ensure probability is JSON serializable
        })

    except FileNotFoundError:
        logging.error(f"Error saving file, possibly invalid path or permissions: {destination}")
        return jsonify({"error": "Error saving uploaded file."}), 500
    except Exception as e:
        logging.error(f"An unexpected error occurred during prediction processing:")
        logging.error(traceback.format_exc()) # Log the full error stack
        return jsonify({"error": f"An unexpected server error occurred."}), 500

@app.route("/api/examples", methods=["GET"])
def get_examples():
    """Returns a list of example image filenames and their URLs."""
    logging.debug("Request received for /api/examples")
    examples = []
    try:
        if not os.path.exists(EXAMPLE_FOLDER_ABS):
             logging.warning(f"Example image directory not found: {EXAMPLE_FOLDER_ABS}")
             return jsonify([]) # Return empty list if directory doesn't exist

        # Sort filenames for consistent order
        filenames = sorted([f for f in os.listdir(EXAMPLE_FOLDER_ABS)
                           if os.path.isfile(os.path.join(EXAMPLE_FOLDER_ABS, f)) and
                           os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS])

        for filename in filenames:
            try:
                examples.append({
                    "filename": filename,
                    # Generate URL using url_for for the static file
                    "url": url_for('static', filename=f"{EXAMPLE_FOLDER_REL}/{filename}", _external=False) # Use relative URL
                })
            except Exception as url_err:
                 logging.error(f"Error generating URL for example '{filename}': {url_err}")
                 # Skip this example if URL can't be generated

        logging.info(f"API returning {len(examples)} example images.")

    except Exception as e:
        logging.error(f"Error listing example images via API: {e}")
        # Return an error response instead of an empty list on general errors
        return jsonify({"error": "Could not retrieve example images due to a server error."}), 500

    return jsonify(examples) # Return list of examples


# --- Main Execution ---

if __name__ == "__main__":
    logging.info("--- Starting Flask Application ---")
    logging.info(f"Application Root (src): {APP_ROOT}")
    logging.info(f"Static Folder: {STATIC_FOLDER}")
    logging.info(f"Upload Target Directory: {UPLOAD_TARGET_ABS}")
    logging.info(f"Example Directory: {EXAMPLE_FOLDER_ABS}")
    logging.info(f"Templates Folder: {os.path.join(APP_ROOT, 'templates')}")
    # Use debug=True for development, set to False for final demo/presentation
    # use_reloader=False can sometimes help with TensorFlow double-initialization issues in debug mode
    app.run(host='0.0.0.0', port=5555, debug=True, use_reloader=True)
    logging.info("--- Flask Application Stopped ---")