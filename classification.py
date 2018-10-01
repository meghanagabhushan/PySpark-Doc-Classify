import os

from pyspark.sql import SparkSession

from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, FloatType

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import col

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF

spark = SparkSession.builder.getOrCreate()

schema = StructType([
    StructField("_c0", StringType()),
    StructField("_c1", StringType())
])

dataset = spark.read.csv('shuffled-full-set-hashed.csv', header=True, schema=schema)
dataset = dataset.where(col("_c1").isNotNull())

regexTokenizer = RegexTokenizer(inputCol="_c1", outputCol="words", pattern="\\W")
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)
label_stringIdx = StringIndexer(inputCol = "_c0", outputCol = "label")

pipeline = Pipeline(stages=[regexTokenizer, hashingTF,idf, label_stringIdx])
pipelineModel = pipeline.fit(dataset)
dataset = pipelineModel.transform(dataset)

trainingData, testData = dataset.randomSplit([0.7, 0.3], seed=100)

lr = LogisticRegression(maxIter=20, regParam=0.01, elasticNetParam=0)
lrModel = lr.fit(trainingData)

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
predictions = lrModel.transform(testData)
acc = evaluator.evaluate(predictions)
print("Accuracy on testset is:", acc)

lrModel.write().overwrite().save("lr_Model")
pipelineModel.write().overwrite().save("pipeline_Model")

print("Stored pipeline and model.")
