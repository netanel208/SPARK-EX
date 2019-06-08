from pyspark import SparkContext
import numpy as np

sc = SparkContext("local[*]", "Simple App")

input_data = sc.parallelize([("hello there", 0), ("hi there", 0), ("go home", 1),
                             ("see you", 1), ("good bye to you", 1)])


def count_sep(_list):
    """Accepts list that separated by keys and return list that each place
    indicate the amount of instances of specific key"""
    tmp = []
    ans = []
    for t in _list:
        tmp.append([i for i in t[1]])
    for l in tmp:
        ans.append(len(l))
    return ans


# mc-> this is the tuple (message, class)
# cw-> this is the tuple (class, word)

"""Do some change in calculate Naive Bayes - 
the classifier is based on a count of words relative to the number of words in the dictionary,
i.e the pk now inside the PI is num_of_words"""
p_k = input_data.map(lambda mc: (mc[1], 1)).reduceByKey(lambda a, b: a+b).collectAsMap()
num_of_words = count_sep(input_data
                         .flatMap(lambda mc: list(list([(mc[1], w) for w in mc[0].split()])))
                         .groupBy(lambda c: c[0])
                         .collect())

#    .map(lambda cw: ((cw[0], cw[1]), 1)) \
#    .reduceByKey(lambda a, b: a+b)\
#    .count()

print("num_of_words", num_of_words)

p_tot = sum(p_k.values())
p_ki = input_data \
    .flatMap(lambda mc: list(set([(mc[1], w) for w in mc[0].split()]))) \
    .map(lambda cw: ((cw[0], cw[1]), 1)) \
    .reduceByKey(lambda a, b: a+b).collectAsMap()

query = "hello hi"
class_probs = [float(p_k[k]+1)/float(p_tot+2)
               * np.prod(np.array([float(p_ki.get((k, i), 0)+1)/float(num_of_words[k]+2)
                                  for i in query.split()]))for k in range(0, 2)]

print(class_probs, end="\n")

y_star = np.argmax(np.array(class_probs))
print(y_star)



