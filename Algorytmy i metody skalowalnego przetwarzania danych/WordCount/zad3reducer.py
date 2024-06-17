#!/usr/lib/python3.8/python
def reducer(data):
    current_word = None
    current_count = 0
    result = []
    for line in data:
        line = line.strip()
        word, count = line.split('\t')
        count = int(count)

        if current_word == word:
            current_count += count
        else:
            if current_word:
                result.append(f"{current_word}\t{current_count}")
            current_word = word
            current_count = count

    if current_word:
        result.append(f"{current_word}\t{current_count}")
    return result
