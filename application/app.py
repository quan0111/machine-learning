from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

MODEL_PATH = r"..\output\model\extra_trees_model.joblib"
ENCODER_PATH = r"..\output\model\country_encoder.joblib"
DATA_PATH = r"..\dataset\Employment_Unemployment_GDP_data.csv"

try:
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    df = pd.read_csv(DATA_PATH)

    df.rename(columns={
        'Country Name': 'Country',
        'Employment Sector: Agriculture': 'Labor_Agriculture',
        'Employment Sector: Industry': 'Labor_Industry',
        'Employment Sector: Services': 'Labor_Services',
        'Unemployment Rate': 'Unemployment_Rate',
        'GDP (in USD)': 'GDP'
    }, inplace=True)

    if "Country_Code" not in df.columns:
        df["Country_Code"] = le.transform(df["Country"].str.strip())

except Exception as e:
    raise RuntimeError(f"Lỗi khi tải mô hình hoặc dữ liệu: {e}")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_all_countries():
    try:
        data = request.get_json()
        year = int(data["year"])
        agri = float(data["labor_agriculture"])
        ind = float(data["labor_industry"])
        serv = float(data["labor_services"])
        unemp = float(data["unemployment_rate"])

        countries = df.drop_duplicates(subset=["Country_Code"])[["Country_Code", "Country"]]

        new_data = pd.DataFrame({
            "Country_Code": countries["Country_Code"].values,
            "Year": [year] * len(countries),
            "Labor_Agriculture": [agri] * len(countries),
            "Labor_Industry": [ind] * len(countries),
            "Labor_Services": [serv] * len(countries),
            "Unemployment_Rate": [unemp] * len(countries)
        })

        preds = model.predict(new_data)
        results = pd.DataFrame({
            "Country": countries["Country"].values,
            "Predicted_GDP_USD": preds
        }).sort_values(by="Predicted_GDP_USD", ascending=False)

        return jsonify({
            "year": year,
            "results": results.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
