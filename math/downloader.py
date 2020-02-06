#!/usr/bin/env python3

import os
import os.path
import re
import shutil
import urllib.request
from pathlib import Path
import webbrowser
import requests
from bs4 import BeautifulSoup

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36')]
urllib.request.install_opener(opener)

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
                print("ERROR: download content", link, flush=True)
                exit(1)
            return
    else:
        status = download_single_file(link, None, destination)
        if not status:
            print("ERROR: direct download", link, flush=True)
            exit(1)


def download_file_google_usercontent(link, destination):
    id = get_id_google_drive(link)
    url = 'https://drive.google.com/file/d/' + id + '/view'
    resp = urllib.request.urlopen(url)
    soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'), features="html.parser")
    title = soup.find("meta", property="og:title")
    imgurl = soup.find("meta", property="og:image")
    # print(soup.prettify())
    print('LINK :' + link + imgurl["content"], title["content"], flush=True)
    status = download_single_file(imgurl["content"], title["content"], destination)
    if not status:
        print("WARNING: img is downloaded", status)
        link = 'https://drive.google.com/u/0/uc?id=' + id + '&export=download'
        status = download_single_file(link, title["content"], destination)
        if not status:
            print("ERROR: content downloaded directly", link, ', ', title["content"])
    return status

def download_single_file(link, name, destination):
    link = link.strip()
    if not name:
        name = link.rsplit('/', 1)[-1]
    filename = os.path.join(destination, name)
    if not os.path.isfile(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        try:
            local_filename, headers = urllib.request.urlretrieve(link, filename)
            if 'image' in headers['Content-Type']:
                print("DELETE IMG:", link, ', ', filename)
                os.remove(filename)
                return False
            else:
                print("DOWNLOADED:", headers, ', ' + link, filename)
            return True
        except Exception as inst:
            print(inst)
            print('Encountered unknown error. Continuing.', flush=True)
            webbrowser.get("open -a /Applications/Google\ Chrome.app %s").open(link)
            #input("continue...")
            return True
    else:
        print('SKIP : ' + link, ', ', name)
        return True


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
        print('Warning: id=' + id)
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

def create_url_list(top_urls_list, dest_file):
    url_lst = []
    for url in top_urls_list:
        resp = urllib.request.urlopen(url)
        soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'), features="html.parser")
        for link in soup.find_all('a', href=True):
            url_lst.append(link['href'])
    f = open(dest_file, 'w')
    f.write('\n'.join(url_lst))
    f.close()
    return url_lst

def load_url_list(src_file):
    return open(src_file).readlines()

def move_files(src_d, dest_d):
    for fname in os.listdir(src_d):
        filename_dir = os.path.dirname(fname)
        filename_w_ext = os.path.basename(fname)
        filename, file_extension = os.path.splitext(filename_w_ext)
        filename = filename.lower().replace(" ", "_")
        shutil.copyfile(os.path.join(src_d, fname), os.path.join(dest_d, filename + file_extension))

if __name__ == "__main__":
    ST_DIR = str(Path.home()) + r'/Downloads'
    patterns = []
    patterns.append('https://drive.google.com')
    # patterns.append('pdf')
    #url_list = fetch_url_list(url, patterns, yes_or_no=True)

    url_list = ['']
    create_url_list(url_list, 'urls.txt')
    download_all(load_url_list('urls.txt'), DST_DIR)
    download_all(url_list, DST_DIR)
    move_files("src", "dst")
    download_file("http://", str(Path.home()) + r'/Downloads')
