import pandas as pd
from pathlib import Path
import tarfile
import urllib.request
from fileExtractor import load_database
import os

# 
print("Hello")
# NOT IN DF2.TSV
#name_basics = load_name_basics()
#title_akas = load_title_akas()

#IN DF2>TSV
# title_basics = load_title_basics()
# title_crew = load_title_crew()
# title_ratings = load_title_ratings()

# This is the code to create df2 once you download all the datasets and put them into a datasets folder
# df1 = title_basics.merge(title_crew, on="tconst")
# df2 = df1.merge(title_ratings,on="tconst")
# df2.to_csv('datasets/df2.tsv', sep='\t', index=False, header=True)
db = load_database()
print(db)