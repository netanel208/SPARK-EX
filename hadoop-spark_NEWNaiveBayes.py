from pyspark import SparkContext
import numpy as np

sc = SparkContext("local[*]", "Simple App")

input_data = sc.parallelize([("hello there", 0), ("hi there", 0), ("go home", 1),
                             ("see you", 1), ("good bye to you", 1)])

# mc-> this is the tuple (message, class)
# cw-> this is the tuple (class, word)

"""Do some change in calculate Naive Bayes - 
the classifier is based on a count of words relative to the number of words in the dictionary,
i.e the pk now inside the PI is num_of_words"""
p_k = input_data.map(lambda mc: (mc[1], 1)).reduceByKey(lambda a, b: a+b).collectAsMap()
num_of_words = input_data \
    .flatMap(lambda mc: list(list([(mc[1], w) for w in mc[0].split()]))) \
    .count()

#    .map(lambda cw: ((cw[0], cw[1]), 1)) \
#    .reduceByKey(lambda a, b: a+b)\
#    .count()

print("num_of_words = ", num_of_words)

p_tot = sum(p_k.values())
p_ki = input_data \
    .flatMap(lambda mc: list(set([(mc[1], w) for w in mc[0].split()]))) \
    .map(lambda cw: ((cw[0], cw[1]), 1)) \
    .reduceByKey(lambda a, b: a+b).collectAsMap()

query = "hello hi"
class_probs = [p_k[k]/float(p_tot)
               * np.prod(np.array([p_ki.get((k, i), 0)/float(num_of_words)
                                  for i in query.split()]))for k in range(0, 2)]

print(class_probs, end="\n")

y_star = np.argmax(np.array(class_probs))
print(y_star)




