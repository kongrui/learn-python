#!/usr/bin/env python3

import os
import os.path
import sys
import time
import urllib.request
import requests
import re

from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PARENT_DIR, 'data')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def save_all_urls(url, patterns, yes_or_no=True):
    url_list = fetch_url_list(url, patterns, yes_or_no=False)
    p = os.path.join(DATA_DIR, 'urls.txt')
    f = open(p, 'w')
    f.writelines(url_list)
    f.close()

def fetch_url_list(url, patterns, yes_or_no=True):
    urllst = []
    resp = urllib.request.urlopen(url)
    soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'), features="html.parser")
    for link in soup.find_all('a', href=True):
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
    return urllst

def download_file_from_google_drive(id):

    def get_file_name_google_drive(response):
        print(response.headers)
        if 'Content-Disposition' in response.headers:
            content_name = response.headers['Content-Disposition']
            content_name_components = re.search(r'filename\=\"(.*)\"', response.headers['Content-Disposition'])
            if content_name_components:
                return content_name_components.group(1)
            else:
                print(content_name)

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    name = get_file_name_google_drive(response)
    if name:
        destination = os.path.join(DATA_DIR, name.replace(" ", "_"))
    else:
        destination = os.path.join(DATA_DIR, id + ".pdf")
    save_response_content(response, destination)

if __name__ == "__main__":
    pass
