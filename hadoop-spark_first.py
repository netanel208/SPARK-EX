from pyspark import SparkConf, SparkContext
from pyspark.sql import Row
from pyspark.sql import SQLContext

from pyspark.sql import functions
from pyspark.sql.functions import lit  # lit for literal


sc = SparkContext("local[*]", "Simple App")
print(sc.textFile("prices.txt").count())

sqlContext = SQLContext(sc)
student_list = [(111, 'Chaya', 'Glass', 21), (222, 'Tal', 'Negev', 28),  	(333, 'Gadi', 'Golan', 24), (444, 'Moti', 'Cohen', 23)]
student_rdd = sc.parallelize(student_list)  # create RDD from python list with using all core(local[*]) parallelize
students_rows = student_rdd.map(lambda x: Row(id=int(x[0]), age=int(x[3]), firstName=x[1], lastName=x[2]))
df_students = sqlContext.createDataFrame(students_rows)
df_students.show()


# READ JSON

students_json = '[{"id":"111","firstName":"Chaya","lastName":"Glass","age":23},' \
                '{"id":"222","firstName":"Tal","lastName":"Negev","age":28},' \
                '{"id":"333","firstName":"Gadi","lastName":"Golan","age":24},' \
                '{"id":"444","firstName":"Moti","lastName":"Cohen","age":23}]'
df = sqlContext.read.json(sc.parallelize([students_json]))  # read->return dataframereader that can be use to read data
df.show()
print("\n")


# ADD COLUMN

df_students2 = df_students.withColumn('young', 	functions.when(df_students.age < 25, True).
                                      otherwise(False)).withColumn('max_grade', lit(100))
df_students2.show()


# MAP REDUCE






