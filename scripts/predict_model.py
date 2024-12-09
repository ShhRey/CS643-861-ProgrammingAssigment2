from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark Session
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

# Load Data
training_data_path = "datasets/TrainingDataset.csv"
validation_data_path = "datasets/ValidationDataset.csv"

train_df = spark.read.csv(training_data_path, header=True, inferSchema=True)
val_df = spark.read.csv(validation_data_path, header=True, inferSchema=True)

# Prepare Data
feature_columns = train_df.columns[:-1]  # Exclude the target column
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
train_df = assembler.transform(train_df).select("features", "quality")
val_df = assembler.transform(val_df).select("features", "quality")

# Train Model
lr = LinearRegression(featuresCol="features", labelCol="quality")
lr_model = lr.fit(train_df)

# Evaluate Model
predictions = lr_model.transform(val_df)
evaluator = RegressionEvaluator(labelCol="quality", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) on validation data: {rmse}")

# Save Model
model_path = "E:/NJIT-MSCS/SEM-3/CloudComputing/ProgAssignment-2/model/wine_quality_model"
lr_model.save(model_path)

print(f"Model saved to {model_path}")

spark.stop()