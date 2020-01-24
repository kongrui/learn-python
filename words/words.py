#!/usr/bin/env python3

import playsound
import time
import sys
import os
import os.path
from bs4 import BeautifulSoup
import urllib.request

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PARENT_DIR, 'data')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

DIR_OUTPUT = os.path.join(DATA_DIR, '{word}{ext}')

URL_WORD = 'https://www.thefreedictionary.com/{word}'
URL_PRONOUCIATION = 'http://img2.tfd.com/pron/mp3/{snd}.mp3'

def save_audio(word, snd):
    url = URL_PRONOUCIATION.format(snd=snd)
    dst = DIR_OUTPUT.format(word=word, ext=".mp3")
    if not os.path.exists(dst):
        urllib.request.urlretrieve(url, dst)
        dst = DIR_OUTPUT.format(word=word, ext=".mp3.txt")
        file = open(dst, "w")
        file.write(url)
        file.close()

def save_definition(word, definition):
    dst = DIR_OUTPUT.format(word=word, ext=".txt")
    if not os.path.exists(dst):
        file = open(dst, "w")
        file.write(definition)
        file.close()

def save_page(word, page, ext):
    ext = ext + ".html"
    dst = DIR_OUTPUT.format(word=word, ext=ext)
    if not os.path.exists(dst):
        file = open(dst, "w")
        file.write(page)
        file.close()


def get_word(word):
    dst_audio = DIR_OUTPUT.format(word=word, ext=".mp3")
    dst_txt = DIR_OUTPUT.format(word=word, ext=".txt")
    if os.path.exists(dst_audio) and os.path.exists(dst_txt):
        return
    page = urllib.request.urlopen(URL_WORD.format(word=word))
    soup = BeautifulSoup(page, from_encoding=page.info().get_param('charset'), features="html.parser")
    snd2_spans = soup.find_all(class_="snd2")
    sndUS = ""
    sndUK = ""
    for snd2_span in snd2_spans:
        snd = snd2_span.attrs['data-snd']
        if "en/US" in snd:
            sndUS = snd
        if "en/UK" in snd:
            sndUK = snd
    if sndUS:
        save_audio(word, sndUS)
    else:
        if sndUK:
            save_audio(word, sndUK)
        else:
            print("ERROR: audio is not found - " + word)
            save_page(word, soup.prettify(), ".mp3")

    definition = soup.find(id="Definition")
    if not definition:
        print("ERROR: definition is not found - " + word)
        save_page(word, soup.prettify(), ".def")
        return

    def_slst = definition.find_all(class_="sds-list")
    txt = ""
    for d in def_slst:
        txt = txt + os.linesep + d.get_text().strip()
    def_single = definition.find_all(class_="ds-single")
    for d in def_single:
        txt = txt + os.linesep + d.get_text().strip()
    def_lst = definition.find_all(class_="ds-list")
    for d in def_lst:
        txt = txt + os.linesep + d.get_text().strip()
    if txt.strip():
        save_definition(word, txt.strip())
    else:
        print("ERROR: details is not found - " + word)
        save_page(word, definition.prettify(), ".def")

def show_word(word):
    f = DIR_OUTPUT.format(word=word, ext=".mp3")
    if os.path.exists(f):
        playsound.playsound(f, True)
    f = DIR_OUTPUT.format(word=word, ext=".mp3.txt")
    if os.path.exists(f):
        print(open(f).read())
    f = DIR_OUTPUT.format(word=word, ext=".txt")
    if os.path.exists(f):
        print(open(f).read())

def count_down(duration):
    for remaining in range(duration, 0, -1):
        sys.stdout.write("\r")
        sys.stdout.write("{:2d} seconds remaining.".format(remaining))
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\rComplete!            \n")

def process(word):
    cmd = input("\nContinue? Yy/Nn/Aa")
    reply = cmd.lower().strip()
    if not reply:
        reply = 'y'
    if reply[0] == 'a':
        exit(0)
    if reply[0] == 'n':
        return False
    try:
        print("--------------------------------")
        print("** " + word + " **")
        print("--------------------------------")
        get_word(word)
        show_word(word)
    except:
        print('now sleeping for 120 seconds')
        count_down(30)

if __name__ == "__main__":
    with open(os.path.join(DATA_DIR, 'words.txt'), 'r') as f:
        for line in f:
            for word in line.split():
                if word:
                    process(word)
