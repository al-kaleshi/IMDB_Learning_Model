from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

#df.loc[['viper', 'sidewinder']]

def load_name_basics():
    # name_basics
    file_name = 'datasets/name.basics.tsv'
    name_basics_df = pd.read_csv(file_name,
                    sep='\t',
                    usecols=[0,1],
                    header=0)
    return name_basics_df

def load_title_basics():
    file_name = 'datasets/title.basics.tsv'
    title_basics_df = pd.read_csv(file_name,
                    sep='\t',
                    usecols=[0,2,5,7,8],
                    header=0)
    return title_basics_df

def load_title_crew():
    file_name = 'datasets/title.crew.tsv'
    title_crew_df = pd.read_csv(file_name,
                    sep='\t',
                    usecols=[0,1,2],
                    header=0)
    return title_crew_df

def load_title_ratings():
    file_name = 'datasets/title.ratings.tsv'
    title_ratings_df = pd.read_csv(file_name,
                    sep='\t',
                    usecols=[0,1],
                    header=0)
    return title_ratings_df

def load_title_akas():
    file_name = 'datasets/title.akas.tsv'
    title_akas_df = pd.read_csv(file_name,
                    sep='\t',
                    usecols=[0,3,7],
                    low_memory=False,
                    header=0)
    return title_akas_df

def load_database():
    file_name = 'datasets/df2.tsv'
    df = pd.read_csv(file_name,
                    sep='\t',
                    low_memory=False,
                    header=0)
    return df
