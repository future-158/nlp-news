import pickle
import os
from pathlib import Path
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import joblib


cfg = OmegaConf.load("conf/config.yaml")

# load bbc dataset
categories = ["politics", "tech", "sport", "business", "entertainment"]
items = []
for category in categories:
    for doc in (Path(cfg.catalog.bbc.raw) / category).glob("*.txt"):
        with open(doc, "r") as f:
            item = {
                "category": category,
                "en_sentence": f.read().strip().replace("\n\n", " "),
                "split": 'appendix'
            }
            items.append(item)

appendix = pd.DataFrame(items)

# merge categories
renamer = {
    "politics": 5, # politics
    "tech": 4,  # it
    "sport": 7, # sport
    "business": 1, # economy
    "entertainment": 2, # entertain
}

appendix.category = appendix.category.map(renamer)

appendix.to_csv(
    cfg.catalog.appendix, index=False
)
