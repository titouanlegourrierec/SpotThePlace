import os
from flask import request, jsonify, render_template
from werkzeug.utils import secure_filename
from .utils import predict_country

def init_routes(app):
    @app.route("/", methods=["GET"])
    def home():
        return render_template("index.html")

    @app.route("/analyze", methods=["POST"])
    def analyze():
        if "image" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        prediction = predict_country(filepath)

        os.remove(filepath)

        return jsonify({"country": prediction})
