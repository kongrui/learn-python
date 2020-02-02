#!/usr/bin/env python3

import os
import os.path
import sys
import time
import urllib.request

#import playsound
from bs4 import BeautifulSoup
import platform

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
    dst_txt = DIR_OUTPUT.format(word=word, ext=".mp3.txt")
    if not os.path.exists(dst_txt):
        file = open(dst, "w")
        file.write(url)
        file.close()


def save_page(word, page_full):
    dst = DIR_OUTPUT.format(word=word, ext=".html")
    if not os.path.exists(dst):
        file = open(dst, "w")
        file.write(page_full)
        file.close()


def get_word_page(word):
    page = urllib.request.urlopen(URL_WORD.format(word=word))
    soup = BeautifulSoup(page, from_encoding=page.info().get_param('charset'), features="html.parser")
    save_page(word, soup.prettify())


def get_word_resource(word):
    dst_page = DIR_OUTPUT.format(word=word, ext=".html")
    if os.path.exists(dst_page):
        return

    get_word_page(word)
    file = open(dst_page, "r")
    soup = BeautifulSoup(file.read(), features="html.parser")
    file.close()
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

    definition = soup.find(id="Definition")
    if not definition:
        print("ERROR: definition is not found - " + word)
        return
    para_defs = definition.find_all(class_="pseg")
    txt = ""
    for one_para in para_defs:
        word_type = one_para.select_one('i')
        if word_type:
            txt = txt + os.linesep + word_type.get_text().strip().replace('\s*\n\s*', '').replace('\s*\r\s*', '')
        def_lst = one_para.find_all(class_="ds-list")
        if def_lst:
            for d in def_lst:
                txt = txt + os.linesep + d.get_text().strip().replace('\s*\n\s*', '').replace('\s*\r\s*', '')
            def_slst = one_para.find_all(class_="sds-list")
            for d in def_slst:
                txt = txt + os.linesep + d.get_text().strip().replace('\s*\n\s*', '').replace('\s*\r\s*', '')

        def_single = one_para.find_all(class_="ds-single")
        for d in def_single:
            txt = txt + os.linesep + d.get_text().strip()

    if txt.strip():
        dst = DIR_OUTPUT.format(word=word, ext=".txt")
        file = open(dst, "w")
        file.write(txt.strip())
        file.close()

def show_word(word):
    f = DIR_OUTPUT.format(word=word, ext=".mp3")
    if os.path.exists(f):
        # C:\Program Files\VideoLAN\VLC to path
        if platform.system() == 'Windows':
            os.system("vlc.exe --qt-start-minimized --play-and-exit " + f)
        else:
            os.system("/Applications/VLC.app/Contents/MacOS/VLC -I rc --play-and-exit " + f)
            #playsound.playsound(f, True)

    f = DIR_OUTPUT.format(word=word, ext=".mp3.txt")
    if os.path.exists(f):
        print(open(f).read())
    f = DIR_OUTPUT.format(word=word, ext=".txt")
    if os.path.exists(f):
        original_text = open(f).read()
        text = os.linesep.join([ll.rstrip() for ll in original_text.splitlines() if ll.strip()])
        print(text)

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
        get_word_resource(word)
        show_word(word)
    except Exception as e:
        print(e)
        print('now sleeping for 10 seconds')
        count_down(10)


if __name__ == "__main__":
    with open(os.path.join(DATA_DIR, 'words.lst'), 'r') as f:
        for line in f:
            for word in line.split():
                if word:
                    word = word.strip()
                    if word.startswith('#'):
                        continue
                    else:
                        process(word)
