import sys
from pyspark.sql import SparkSession


def map(line):
    import nltk
    import string
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk.download('stopwords')
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    cleaned_line = line.translate(str.maketrans('', '', string.punctuation)).lower()
    stop_words = set(stopwords.words('english'))
    words = [lemmatizer.lemmatize(word) for word in cleaned_line.split() if word not in stop_words]
    return [(w, 1) for w in words]


def reducer(data):
    key, values = data
    return (key, sum(values))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: wordcount <file>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession \
        .builder \
        .appName("PythonWordCount") \
        .getOrCreate()

    lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])

    countsR = lines.flatMap(map).groupByKey().map(reducer).sortBy(lambda x: x[1], ascending=True)

    for word, count in countsR.collect():
        print("%s: %s" % (word, count))

    spark.stop()
