import os
import json

# Starting the Spark Session 
from pyspark.sql import SparkSession 

    # Importing the required libraries 
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder 

# Importing Pipeline and Model 
from pyspark.ml import Pipeline 
from pyspark.ml.classification import LogisticRegression 

# Importing the evaluator 
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def train():
    try:
        spark = SparkSession.builder.appName('MusicLyricsModel').getOrCreate() 
        training_csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.path.join(os.getenv('TRAIN_DATA_DIR'), os.getenv('TRAIN_DATA_CSV')))
        # Reading the data 
        df = spark.read.option("maxRowsInMemory", 1000000).csv(training_csv_path, inferSchema=True, header=True) 
        
        rm_columns = df.select(['artist_name',
            'track_name',
            'release_date',
            'genre',
            'lyrics',
            'len',
            'dating',
            'violence',
            'world/life',
            'night/time',
            'shake the audience',
            'family/gospel',
            'romantic',
            'communication',
            'obscene',
            'music',
            'movement/places',
            'light/visual perceptions',
            'family/spiritual',
            'like/girls',
            'sadness',
            'feelings',
            'danceability',
            'loudness',
            'acousticness',
            'instrumentalness',
            'valence',
            'energy',
            'topic',
            'age']) 

        # Drops the data having null values 
        result = rm_columns.na.drop() 
        

        # Converting the genre Column 
        genreIdx = StringIndexer(inputCol='genre', 
                                    outputCol='GenreIndex') 
        genreEncode = OneHotEncoder(inputCol='GenreIndex', 
                                    outputCol='GenreVec') 

        # Converting the lyrics Column 
        lyricsIdx = StringIndexer(inputCol='lyrics', 
                                    outputCol='LyricsIndex').setHandleInvalid("keep") 
        lyricsEncode = OneHotEncoder(inputCol='LyricsIndex', 
                                    outputCol='LyricsVec') 

        # Converting the artist_name Column 
        artist_nameIdx = StringIndexer(inputCol='artist_name', 
                                    outputCol='artist_nameIndex').setHandleInvalid("keep") 
        artist_nameEncode = OneHotEncoder(inputCol='artist_nameIndex', 
                                    outputCol='artist_nameVec') 

        # Converting the track_name Column 
        track_nameIdx = StringIndexer(inputCol='track_name', 
                                    outputCol='track_nameIndex').setHandleInvalid("keep") 
        track_nameEncode = OneHotEncoder(inputCol='track_nameIndex', 
                                    outputCol='track_nameVec') 

        # Converting the topic Column 
        topicIdx = StringIndexer(inputCol='topic', 
                                    outputCol='topicIndex').setHandleInvalid("keep") 
        topicEncode = OneHotEncoder(inputCol='topicIndex', 
                                    outputCol='topicVec') 

        # Vectorizing the data into a new column "features" 
        # which will be our input/features class 
        assembler = VectorAssembler(inputCols=['artist_nameVec',
            'track_nameVec',
            'release_date',
            'GenreVec',
            'LyricsVec',
            'len',
            'dating',
            'violence',
            'world/life',
            'night/time',
            'shake the audience',
            'family/gospel',
            'romantic',
            'communication',
            'obscene',
            'music',
                'movement/places',
                'light/visual perceptions',
                'family/spiritual',
                'like/girls',
                'sadness',
                'feelings',
                'danceability',
                'loudness',
                'acousticness',
                'instrumentalness',
                'valence',
                'energy',
                'topicVec',
                'age'], outputCol='features') 
        
        log_reg = LogisticRegression(featuresCol='features', 
                                labelCol='GenreIndex') 

        # Creating the pipeline 
        pipe = Pipeline(stages=[genreIdx, lyricsIdx, artist_nameIdx, track_nameIdx, topicIdx,
                                    genreEncode, lyricsEncode, artist_nameEncode, track_nameEncode, topicEncode, 
                                    assembler, log_reg]) 
        
        # Splitting the data into train and test 
        train_data, test_data = result.randomSplit([0.8, .2]) 

        # Fitting the model on training data 
        fit_model = pipe.fit(train_data) 
        
        # Extract the genre indices from the trained model
        stringIndexerModel = fit_model.stages[0]
        labels = stringIndexerModel.labels
        genre_mapping = {index: label for index, label in enumerate(labels)}
        
        # save the genre mapping for use in prediction
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), os.path.join(os.getenv('MODEL_DIR'), os.getenv('GENRE_MAPPING_FILE'))), 'w') as f:
            f.write(json.dumps(genre_mapping))
        
        # save the trained model
        # Instead of using save_obj, directly save the model using Spark's save method
        fit_model.write().overwrite().save(os.path.join(os.path.dirname(os.path.dirname(__file__)), os.path.join(os.getenv('MODEL_DIR'), os.getenv('MODEL_FILE'))))

        # Storing the results on test data 
        results = fit_model.transform(test_data) 
        
        # Calling the evaluator 
        evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='GenreIndex') 

        # Evaluating the AUC on results 
        ROC_AUC = evaluator.evaluate(results) 
        
        return f"Model accuracy: {ROC_AUC}"
    except Exception as e:
        print(e)
        return f"Error training model: {e}"


