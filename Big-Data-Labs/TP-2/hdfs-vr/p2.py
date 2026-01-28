from pyspark import SparkConf, SparkContext

nomappli = "p2"
config = SparkConf() \
    .setAppName(nomappli) 

sc = SparkContext(conf=config)

# load the file
file = sc.textFile("hdfs:///user/root/arbres.csv")

# get the header
header = file.first()

# remove the header
file_without_header = file.filter(lambda row: row != header)

# get the columns
columns = header.split(";")

# get the index of the desired column
idx = columns.index("HAUTEUR")

# 1- split the rows
# 2- project the desired column
# 3- remove elements that can not be converted to float
# 4- convert the remaining elements to float
hauteur = file_without_header \
    .map(lambda row : row.split(";")) \
    .map(lambda row : row[idx]) \
    .filter(lambda row : row.replace(".","").isnumeric()) \
    .map(lambda row : float(row))

# compute the mean
mean = hauteur.mean()

print("\n\n------- mean,size = ", (mean,hauteur.count()), " -------\n\n")