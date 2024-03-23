import os
import json

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

def predict(lyrics: str):
    try:
        # Load the model using Spark's load method
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.path.join(os.getenv('MODEL_DIR'), os.getenv('MODEL_FILE')))
        model = PipelineModel.load(model_path)
        
        # Initialize Spark session
        spark = SparkSession.builder.appName('MusicLyricsPredictor').getOrCreate()
        
        # Create a DataFrame with the input lyrics (assuming 'lyrics' is the column expected by the model)
        df = spark.createDataFrame([(lyrics,)], ['lyrics'])

        # Predict
        predictions = model.transform(df)
        
        # Convert numerical index to genre name
        predicted_genre_index = predictions.select('prediction').collect()[0]['prediction']
        genre_mapping_json = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), os.path.join(os.getenv('MODEL_DIR'), os.getenv('GENRE_MAPPING_FILE'))), 'r')
        genre_mapping = json.load(genre_mapping_json)
        predicted_genre = genre_mapping.get(predicted_genre_index, "Unknown Genre")
        
        genre_mapping.close()

        return predicted_genre
    except Exception as e:
        print(e)
        return f'Error predicting genre: {e}'

