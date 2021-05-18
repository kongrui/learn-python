#!/usr/bin/env python3

import dotenv
import os
import os.path
import re
import shutil
import urllib.request
from pathlib import Path
import webbrowser
import requests
from bs4 import BeautifulSoup
import time

def fetch_url_list(url, patterns, yes_or_no=True, fpath=None):
    urllst = []
    resp = urllib.request.urlopen(url)
    soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'), features="html.parser")
    links = []
    for link in soup.find_all('a', href=True):
        links.append(link['href'])
        if yes_or_no:
            for pattern in patterns:
                if pattern in link['href']:
                    urllst.append(link['href'])
        else:
            added = True
            for pattern in patterns:
                if pattern in link['href']:
                    added = False
            if added:
                urllst.append(link['href'])
    if fpath:
        f = open(fpath, 'w')
        f.write('\n'.join(links))
        f.close()
    return urllst

WS_DIR = str(Path.home()) + r'/Downloads'

if __name__ == "__main__":

    f = open('.out.tex.json', 'w')
    root = []
    row_cnt = 0
    dotenv.load_dotenv()
    URL = os.getenv("EP_URL")
    resp = urllib.request.urlopen(URL)
    soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'), features="html.parser")
    probs = soup.find_all(id=lambda x: x and x.startswith('Problem_'))
    for prob in probs:
        print('\n')
        print(prob['id'])
        prob_tag = prob.parent.parent
        p_tags = prob_tag.find_all('p')
        for tag in p_tags:
            print(tag.text)
            latex_tag = tag.find('img', {"class": "latex"})
            if latex_tag:
                print(latex_tag['alt'])
        break
    f.close()