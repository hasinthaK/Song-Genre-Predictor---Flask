import os
import numpy as np

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import Tokenizer, HashingTF, IDF

# Initialize Spark session
spark = SparkSession.builder.appName('MusicLyricsPredictor').getOrCreate()

# Function to classify lyrics
def classify_lyrics(lyrics):
    # Load the model using Spark's load method
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.path.join(os.getenv('MODEL_DIR'), os.getenv('MODEL_FILE')))
    model = PipelineModel.load(model_path)

    # Tokenize lyrics
    tokenizer = Tokenizer(inputCol="lyrics", outputCol="inputWords")
    words_df = spark.createDataFrame([(lyrics,)], ["lyrics"])
    words = tokenizer.transform(words_df)

    # Feature engineering (using TF-IDF)
    hashingTF = HashingTF(inputCol="inputWords", outputCol="inputFeatures")
    idf = IDF(inputCol="inputFeatures", outputCol="tfidf_features")
    features = idf.fit(hashingTF.transform(words)).transform(hashingTF.transform(words))

    # Predict genre
    predictions = model.transform(features)
    genre = predictions.select("predicted_genre").collect()[0][0]

    return genre


def predict(lyrics: str):
    try:
        predicted_genre = classify_lyrics(lyrics)
        print("Predicted Genre:", predicted_genre)
        
        return predicted_genre
    except Exception as e:
        print(e)
        return f'Error predicting genre: {e}'

