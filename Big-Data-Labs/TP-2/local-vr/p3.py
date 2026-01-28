from pyspark import SparkConf, SparkContext

nomappli = "p3"
config = SparkConf() \
    .setAppName(nomappli)

sc = SparkContext(conf=config)

### load the file
file = sc.textFile("file:///root/arbres.csv")

### get the header
header = file.first()

### remove the header
file_without_header = file.filter(lambda row: row != header)

### get the columns
columns = header.split(";")

### get the index of the desired columns
genre = columns.index("GENRE")
hauteur = columns.index("HAUTEUR")

### construct key-value pairs

# 1- split the row
# 2- construct key-value pairs
# 3- remove the heights that are not convertible to float
# 4- convert the keys to float
pairs = file_without_header \
    .map(lambda row : row.split(";")) \
    .map(lambda row: (row[hauteur], row[genre])) \
    .filter(lambda row : row[0].replace(".","").isnumeric()) \
    .map(lambda row : (float(row[0]), row[1]))

### sort the pairs by key
sorted_pairs = pairs.sortByKey(ascending=False)

### get the first pair
first_pair = sorted_pairs.first()

### print out the result
print("\n\n --- pair = ", first_pair, " --- \n\n")
