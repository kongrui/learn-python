#!/usr/bin/env python3

import csv
import glob
import os.path

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    fileNames = glob.glob(os.path.join(BASE_DIR, '*-words.csv'))
    word_dict = dict()
    for fname in fileNames:
        csv_reader = csv.reader(open(fname), delimiter=',')
        line_count = 0
        for row in csv_reader:
            k = row[0].lower()
            v = row[1].lower()
            if not k in word_dict:
                word_dict[k] = v

print(word_dict)
with open('output.2.csv', mode='w') as allwords_file:
    csvwriter = csv.writer(allwords_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for key, value in word_dict.items():
        csvwriter.writerow([key, value])
