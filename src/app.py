from flask import Flask, request, render_template

# defined imports
from routes.predict_genre import predict
from routes.train_model import train

# Application code-------------------
app = Flask(__name__)

# Endpoint definitions
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['GET'])
def train_model():
    train()
    return 'Training model'

@app.route('/predict', methods=['POST'])
def predict_genre_from_lyrics():
    lyrics = request.form['lyrics']
    template_vars = {
        'prediction': predict(lyrics),
        'lyrics': lyrics
    }
    
    return render_template('prediction_results.html', **template_vars)

# Start server
app.run(debug=app.config['DEBUG'])

