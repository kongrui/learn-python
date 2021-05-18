#!/usr/bin/env python3

import glob
import os.path

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    all_words = dict()
    os.chdir(SCRIPT_DIR)
    for fname in glob.glob("cty*.csv"):
        fpath = os.path.join(SCRIPT_DIR, fname)
        f = open(fpath)
        content = f.readlines()
        for x in content:
            word, sec, *rest = x.strip().split(',')
            word = word.lower()
            meaning = sec + ','.join(rest)
            meaning = meaning.strip(',').strip(' ').replace('"', "")
            if not word in all_words:
                all_words[word] = meaning

    all_words_list = sorted(all_words.items())
    # my_df = pd.DataFrame(all_words_list)
    # my_df.to_csv('output.csv', index=False, header=False)
    print(','.join(sorted(all_words.keys())))
