import pandas as pd
import joblib
import tqdm
from functools import partial
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from typing import List
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
import time
from concurrent.futures import ThreadPoolExecutor
import threading


cfg = OmegaConf.load('conf/config.yaml')

train = pd.read_csv(cfg.catalog.raw.train)
test = pd.read_csv(cfg.catalog.raw.test)

cat = pd.concat(
    [
     train.assign(split='train', order =lambda x: x.index),
     test.assign(split='test', order = lambda x: x.index),
    ], ignore_index=True)


cat['mergeBody'] = cat['title'] +' ' + cat['cleanBody']
id_vars = ['split', 'order','category']
# value_vars = ['title','cleanBody']
value_vars = ['mergeBody']
melt = cat.melt(
    id_vars=id_vars,
    value_vars=value_vars,
)

melt['translated'] = ''
melt = melt.set_index(id_vars)

language_code = "te"



chrome_options = Options()
#chrome_options.add_argument("--disable-extensions")
#chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--headless")
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

total = len(melt)
quotas = np.array_split(np.arange(total), 30)


def handle_job(qouta: List[int], l: threading.Lock):
    def handle_tup(tup):
        if tup.translated != '':
            return
        driver.find_element_by_xpath("//textarea[@aria-label='Source text']").clear() 
        time.sleep(3)
        driver.find_element_by_xpath("//textarea[@aria-label='Source text']").send_keys(tup.value) 
        time.sleep(3)
        spans = driver.find_elements_by_xpath("//span[@lang='en']/span")
        trt_sentence = ''.join([span.text for span in spans])
        with l:
            melt.loc[tup.Index, 'translated'] = trt_sentence

    while 1:
        queue = melt.iloc[qouta]
        done = (queue.translated != '').all()
        if done:
            return
        
        try:
            driver = webdriver.Chrome('chromedriver', chrome_options=chrome_options)
            driver.get('https://translate.google.com/#view=home&op=translate&sl=ko&tl=en')
            
            for tup in tqdm.tqdm(queue.itertuples()):
                handle_tup(tup)
        except Exception:
            pass


l = threading.Lock()
with ThreadPoolExecutor(30) as executor:
    futures = executor.map(partial(handle_job, l=l), quotas)
    _ = [f.result() for f in futures]


usecols = ['split', 'order', 'category', 'value', 'translated']
renamer = {
    'split': 'split',
    'order': 'order',
    'category': 'category',
    'value': 'ko_sentence',
    'translated': 'en_sentence'
    }

(
    melt
    .reset_index()
    [usecols]
    .rename(columns=renamer)
).to_csv(cfg.catalog.translated, index=False)

 