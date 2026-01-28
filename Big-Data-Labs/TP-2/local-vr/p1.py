from pyspark import SparkConf, SparkContext

nomappli = "p1"
config = SparkConf() \
    .setAppName(nomappli) 

sc = SparkContext(conf=config)

data = sc.textFile("file:///root/arbres.csv")

# get the header
header = data.first()

# remove the header
file_without_header = data.filter(lambda row: row != header)

print("\n\n --- size = ",file_without_header.count()," --- \n\n")
