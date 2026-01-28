### Necessary Imports
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import StringType,IntegerType,StructField,StructType

### Setup
nomappli = "main"
config = SparkConf().setAppName(nomappli)
sc = SparkContext(conf=config)

sqlContext = SQLContext(sc)

### Data Loading
file = sc.textFile("hdfs:///user/root/ngram.csv")
table = file \
  .map(lambda row: row.split("\t")) \
  .map(lambda row: (row[0], int(row[1]),int(row[2]),int(row[3]),int(row[4])))

### Schema definition
ngram = StructField("ngram", StringType())
year = StructField("year", IntegerType())
count = StructField("count", IntegerType())
pages = StructField("pages", IntegerType())
books = StructField("books", IntegerType())

schema = StructType([ngram,year,count,pages,books])

### Create a dataframe
ngram = sqlContext.createDataFrame(data=table, schema=schema)

### Register TempTable
ngram.registerTempTable("ngram")

### Queries

# <--- Question 01 : --->

print("\n\nReturn all the bigrams where the the count is greater than 5\n\n")

print("Using SQL : ")

sqlContext.sql("""
  SELECT DISTINCT ngram from ngram as n
  WHERE n.count > 5
""").show()

print("Using PySQL : ")

ngram.filter("count > 5") \
  .select(ngram.ngram) \
  .distinct() \
  .show()

# <--- Question 02 : --->

print("\n\nReturn the number of bigrams for each year\n\n")

print("Using SQL : ")

sqlContext.sql("""
  SELECT n.year,count(n.ngram) as count from ngram as n
  GROUP BY n.year
""").show()

print("Using PySQL : ")

ngram \
  .groupBy("year") \
  .count() \
  .show()

# <--- Question 03 : --->

print("\n\nReturn the bigrams with highest count each year\n\n")

print("Using SQL : ")

sqlContext.sql("""
  SELECT ngram,year,max(count) as count FROM ngram as n
  GROUP BY ngram,year
  ORDER BY year
""").show()

print("Using PySQL : ")

ngram \
  .groupBy(["ngram",'year']) \
  .max("count") \
  .sort(["year"]) \
  .show()


# <--- Question 04 : --->

print("\n\nReturn the bigrams that appeared in 20 different year.\n\n")

print("Using SQL : ")

sqlContext.sql("""
  SELECT ngram,COUNT(DISTINCT year) as years_count
  FROM ngram as n
  GROUP BY ngram
  HAVING years_count >= 20
""").show()

print("Using PySQL : ")

ngram \
  .groupBy(ngram.ngram) \
  .count() \
  .filter("count >= 20") \
  .show()

# <--- Question 05 : --->

print("\n\nReturn the bigrams where `!` is the first character and `9` is the second (separated by white space).\n\n")

print("Using SQL : ")

sqlContext.sql("""
  SELECT DISTINCT ngram FROM ngram as n
  WHERE n.ngram LIKE '%!% %9%'
""").show()

print("Using PySQL : ")

ngram \
  .filter(ngram.ngram.like('%!% %9%')) \
  .select(ngram.ngram) \
  .distinct() \
  .show()

# <--- Question 06 : --->

print("\n\nReturn the bigrams that appears in all the years\n\n")

print("Using SQL : ")

sqlContext.sql("""
  SELECT ngram,count(DISTINCT year) as years_count FROM ngram as n
  GROUP BY ngram
  HAVING  years_count = (SELECT count(DISTINCT year) as years_count FROM ngram)
""").show()

print("Using PySQL : ")

ngram.select(ngram.ngram,ngram.year) \
  .groupBy(ngram.ngram) \
  .count() \
  .filter(col('count') == ngram.select(ngram.year).distinct().count()) \
  .show()

# <--- Question 07 : --->

print("\n\nReturn the total number of pages & books for each bigram by year in alphabetical order\n\n")

print("Using SQL : ")

sqlContext.sql("""
  SELECT ngram,SUM(pages),SUM(books),year
  FROM ngram as n
  GROUP BY year,ngram
  ORDER BY ngram DESC
""").show()

print("Using PySQL : ")

ngram \
  .select('ngram','year','books','pages') \
  .groupBy('year','ngram') \
  .agg(sum('books').alias('total_books'),sum('pages').alias('total_pages')) \
  .sort(desc('ngram')) \
  .show()

# <--- Question 08 : --->

print("\n\nReturn the number of bigrams by year ordered by year\n\n")

print("Using SQL : ")

sqlContext.sql("""
  SELECT year,count(DISTINCT ngram) as count
  FROM ngram as n
  GROUP BY year
  ORDER BY year DESC
""").show()

print("Using PySQL : ")

ngram \
  .select('year','ngram') \
  .groupBy('year') \
  .count() \
  .sort(desc('year')) \
  .show()