#!/usr/bin/env python3

import os
import os.path
import re
import urllib.request
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DST_DIR = str(Path.home()) + r'/Downloads/downthemall'


def fetch_url_list(url, patterns, yes_or_no=True):
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
    f = open('f.all.lst', 'w')
    f.write('\n'.join(links))
    f.close()
    return urllst


def gdown_all(links, destination):
    for link in links:
        id = get_id_google_drive(link)
        if id:
            print('Downloading: ' + link)
            os.system('gdown https://drive.google.com/uc?id=' + id)
        else:
            print("ERROR -  GDOC : id is not found")

def download_all(links, destination):
    for link in links:
        download_file(link, destination)

def download_file(link, destination):
    if "drive.google.com" in link:
        status = download_file_google_drive(link, destination)
        if status:
            pass
        else:
            status = download_file_google_usercontent(link, destination)
            if status:
                pass
            else:
                print("ERROR:", link, flush=True)
            return
    else:
        status = download_single_file(link, None, destination)
        if not status:
            print("ERROR:", link, flush=True)


def download_file_google_usercontent(link, destination):
    id = get_id_google_drive(link)
    url = 'https://drive.google.com/file/d/' + id + '/view'
    resp = urllib.request.urlopen(url)
    soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'), features="html.parser")
    title = soup.find("meta", property="og:title")
    imgurl = soup.find("meta", property="og:image")
    # print(soup.prettify())
    print('LINK :' + link + imgurl["content"], title["content"], flush=True)
    return download_single_file(imgurl["content"], title["content"], destination)


def download_single_file(link, name, destination):
    link = link.strip()
    if not name:
        name = link.rsplit('/', 1)[-1]
    filename = os.path.join(destination, name)
    if not os.path.isfile(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        try:
            urllib.request.urlretrieve(link, filename)
            return True
        except Exception as inst:
            print(inst)
            print('Encountered unknown error. Continuing.')
            return False


def get_id_google_drive(link):
    p = re.compile("drive.google.com.open.id.(.*)")
    m = p.search(link)
    id = None
    if m:
        id = m.group(1)
    else:
        p = re.compile("drive.google.com.file.d.(.*).view?")
        m = p.search(link)
        if m:
            id = m.group(1)

    print('LINK :', link, ' ID=', id, flush=True)
    return id


def download_file_google_drive(link, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    id = get_id_google_drive(link)
    response = session.get(URL, params={'id': id}, stream=True)

    name = get_file_name_google_drive(response)

    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
        name = get_file_name_google_drive(response)

    if not name:
        print('Error with id=' + id)
        return False
    else:
        dest_file = destination + '/' + name
        if not os.path.isfile(dest_file):
            print('Downloading: id=' + id + ", name=" + name, flush=True)
            save_response_content(response, dest_file)
        else:
            print('Skip id=' + id + ", name=" + name, flush=True)
        return True


def get_file_name_google_drive(response):
    if 'Content-Disposition' in response.headers:
        content_name = response.headers['Content-Disposition']
        content_name_components = re.search(r'filename\=\"(.*)\"', response.headers['Content-Disposition'])
        if content_name_components:
            return content_name_components.group(1)
        else:
            print(content_name)
    else:
        print(response.headers)
    return None


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


if __name__ == "__main__":
    patterns = []
    patterns.append('https://drive.google.com')
    # patterns.append('pdf')
    url_list = fetch_url_list(url, patterns, yes_or_no=True)
    f = open('f.lst', 'w')
    f.write('\n'.join(url_list))
    f.close()
    download_all(url_list, DST_DIR)
