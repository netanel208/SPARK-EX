from pyspark import SparkContext
import numpy as np

sc = SparkContext("local[*]", "Simple App")


# def add1(mrd):
#    mrd.update((x, y+1) for x, y in mrd.items())
#    return mrd


input_data = sc.parallelize([("Buy it , pay later ! Click me !", 0), ("you Won 10000 Dollars ! Click here !", 0),
                             ("Are you pay too much ? Click now !", 0), ("Are you pay too much ? Click now !", 1),
                             ("", 0), ("", 1),
                             ("Will See you later .", 1), ("Will you want to meet later ?", 1),
                             ("I am waiting for you .", 1)])

# mc-> this is the tuple (message, class)
# cw-> this is the tuple (class, word)

p_k = input_data.map(lambda mc: (mc[1], 1)).reduceByKey(lambda a, b: a+b).collectAsMap()
p_tot = sum(p_k.values())-2
p_ki = input_data\
    .flatMap(lambda mc: list(set([(mc[1], w) for w in mc[0].split()])))\
    .map(lambda cw: ((cw[0], cw[1]), 1))\
    .reduceByKey(lambda a, b: a+b)\
    .collectAsMap()

print(p_ki)

query = "Are you pay too much ? Click now !"
class_probs = [float(p_k[k]-1)/float(p_tot)
               * np.prod(np.array([float(p_ki.get((k, i), 0))/float(p_k[k])
                                  for i in query.split()]))for k in range(0, 2)]

print(class_probs)

y_star = np.argmax(np.array(class_probs))
print(y_star)




