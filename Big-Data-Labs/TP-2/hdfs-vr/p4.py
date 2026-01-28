from pyspark import SparkConf, SparkContext
from operator import add

nomappli = "p4"
config = SparkConf() \
    .setAppName(nomappli)

sc = SparkContext(conf=config)

### load the file
file = sc.textFile("hdfs:///user/root/arbres.csv")

### get the header
header = file.first()

### remove the header
file_without_header = file.filter(lambda row: row != header)

### get the columns
columns = header.split(";")

### get the index of the desired columns
genre = columns.index("GENRE")

### construct key value pairs (map)
# 1- split the rows
# 2- project the desired column
pairs = file_without_header \
    .map(lambda row : row.split(";")) \
    .map(lambda row : (row[genre], 1)) 

### reduce 
reduced_pairs = pairs.reduceByKey(add)

### print the results
print("\n\n"+str(reduced_pairs.collect())+"\n\n")
