# spark = SparkSession.builder.appName("WineQualityTraining").getOrCreate()
# data = spark.read.csv("datasets/TrainingDataset.csv", header=True, inferSchema=True)

# assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")
# data = assembler.transform(data).select("features", "quality")

# lr = LinearRegression(featuresCol="features", labelCol="quality")
# model = lr.fit(data)
# # path for server: /home/ec2-user/wine_prediction_model/model/wine_quality_model
# model.save("E:/NJIT-MSCS/SEM-3/CloudComputing/ProgAssignment-2/model/wine_quality_model")
# print("Model saved")
# spark.stop()

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Initialize Spark Session
spark = SparkSession.builder.appName("WineQualityTraining").getOrCreate()

# Load Training Data
data = spark.read.csv(
    "datasets/TrainingDataset.csv",
    header=True,          # File contains column headers
    inferSchema=True,     # Infer data types automatically
    sep=";",              # Specify the correct delimiter
    quote='"',            # Handle quoted strings
    multiLine=True        # Handle multi-line records if any
)

# Validate Columns
data.printSchema()
data.show(5)