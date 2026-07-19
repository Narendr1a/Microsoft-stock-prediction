from flask import Flask, request, jsonify, render_template_string

from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.predict_pipeline import PredictPipeline

application = Flask(__name__)
app = application

HOME_PAGE = """
<!DOCTYPE html>
<title>Time series Forecast</title>
<h2>Time series forecasting API</h2>
<p>Endpoints:</p>
<ul>
  <li><b>POST /train</b> — runs the full training pipeline on data in config.yaml's raw_data_path</li>
  <li><b>GET /predict?steps=30</b> — returns a forecast for the given number of steps (default: config value)</li>
</ul>
"""

@app.route("/")
def home():
    return render_template_string(HOME_PAGE)

@app.route("/train", methods = ["POST"])
def train():
    try:
        pipeline = TrainPipeline()
        results = pipeline.run()
        return jsonify({
            "status": "success",
            "best_order": results["best_order"],
            "best_seasonal_order": results["best_seasonal_order"],
            "metrics": results["metrics"],
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/predict", methods = ["GET"]) 
def predict():
    try: 
        steps = request.args.get("steps", default = None, type = int)
        predictor = PredictPipeline()
        forecast_series = predictor.forecast(steps = steps)

        result = {
            str(date.date()):float(value)
            for date, value in forecast_series.items()
        }
        return jsonify({"status": "success", "forecast": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)