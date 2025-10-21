from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# --- Đường dẫn mô hình ---
MODEL_PATH = r"..\output\model\extra_trees_model.joblib"

# --- Tải mô hình ---
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Lỗi khi tải mô hình: {e}")

@app.route("/")
def home():
    return jsonify({
        "message": "GDP Prediction API (Flask Version)",
        "status": "running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # --- Kiểm tra đầu vào ---
        required = ["year", "labor_agriculture", "labor_industry", "labor_services", "unemployment_rate"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Thiếu tham số '{field}'"}), 400

        # --- Tạo DataFrame đầu vào ---
        new_data = pd.DataFrame({
            "Year": [data["year"]],
            "Labor_Agriculture": [data["labor_agriculture"]],
            "Labor_Industry": [data["labor_industry"]],
            "Labor_Services": [data["labor_services"]],
            "Unemployment_Rate": [data["unemployment_rate"]],
        })

        # --- Dự đoán ---
        prediction = float(model.predict(new_data)[0])

        return jsonify({
            "year": data["year"],
            "predicted_GDP_USD": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
