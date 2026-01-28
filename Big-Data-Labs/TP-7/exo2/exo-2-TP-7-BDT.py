from pyspark.sql import SparkSession
from time import sleep

spark = SparkSession.builder.appName("Chapitre4").getOrCreate()
static = spark.read.json("./data")

dataSchema = static.schema
static.printSchema()

streaming = spark.readStream.schema(dataSchema).option("maxFilesPerTrigger", 1).json("./data")

activityCounts = streaming.groupBy("gt").count()

activityQuery = activityCounts.writeStream.queryName("activity_counts") \
    .format("memory").outputMode("complete") \
    .start()

spark.streams.active

for x in range(10):
    spark.sql("SELECT * FROM activity_counts").show()
    sleep(2)