from flask import Flask, request, jsonify, render_template_string
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


def train_model():
	data = load_iris()
	X, y = data.data, data.target
	pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=50, random_state=42))
	pipeline.fit(X, y)
	return pipeline, data.target_names


MODEL, TARGET_NAMES = train_model()

# Use a fixed sample from the sklearn dataset (no user input required)
data = load_iris()
SAMPLE_FEATURES = data.data[0].tolist()
SAMPLE_TRUE = int(data.target[0])
SAMPLE_PRED = MODEL.predict([SAMPLE_FEATURES])[0]
SAMPLE_PROBS = MODEL.predict_proba([SAMPLE_FEATURES])[0].tolist()


@app.route('/', methods=['GET'])
def index():
	return jsonify({
		'message': 'Simple ML app using internal dataset sample (no user input).',
		'sample_features': SAMPLE_FEATURES,
		'sample_true_label': int(SAMPLE_TRUE),
		'sample_true_name': TARGET_NAMES[SAMPLE_TRUE],
		'sample_pred_label': int(SAMPLE_PRED),
		'sample_pred_name': TARGET_NAMES[SAMPLE_PRED],
		'sample_probabilities': SAMPLE_PROBS,
	})


if __name__ == '__main__':
	# Run with: python app.py
	app.run(host='0.0.0.0', port=5000, debug=True)

