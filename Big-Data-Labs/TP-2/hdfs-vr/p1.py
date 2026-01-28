from pyspark import SparkConf, SparkContext

nomappli = "p1"
config = SparkConf() \
    .setAppName(nomappli) 

sc = SparkContext(conf=config)

data = sc.textFile("hdfs:///user/root/arbres.csv")

print("\n\n --- size = ",data.count()," --- \n\n")