from election_pdf_analyzer import (
    extract_pdf_metadata,
    convert_pdf_to_images,
    process_all_images,
)
from enhanced_corruption_analysis import (
    load_voter_data,
    detect_voter_id_anomalies,
)
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import dotenv

dotenv.load_dotenv()  # Load environment variables from .env file

# import your existing logic

app = Flask(__name__)
CORS(app)  # allow Next.js frontend to call Flask

# --- Route 1: Upload & analyze a voter data JSON ---


@app.route("/api/analyze-json", methods=["POST"])
def analyze_json():
    try:
        data = request.json  # raw JSON dict from frontend
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        df = load_voter_data(data)
        print(df)
        if df.empty:
            return jsonify({"error": "Failed to load voter data"}), 400

        results = detect_voter_id_anomalies(df)

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Route 2: Upload PDF & extract voter data ---


@app.route("/api/analyze-pdf", methods=["POST"])
def analyze_pdf():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        upload_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(upload_path)

        # Step 1: Metadata
        metadata = extract_pdf_metadata(upload_path)

        # Step 2: Convert to images (skip first 2 pages)
        image_paths, output_folder = convert_pdf_to_images(
            upload_path, start_page=3)

        if not image_paths:
            return jsonify({"error": "Failed to convert PDF to images"}), 500

        # Step 3: Run Groq OCR
        processed_result = process_all_images(
            image_paths, output_folder, os.getenv("GROQ_API_KEY"))

        return jsonify({
            "metadata": metadata,
            "images_processed": len(image_paths),
            "total_voters": processed_result["total_voters"],
            "pages": processed_result["pages"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
