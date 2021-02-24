#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function-based pipeline for feature dataframe preparation using the GTZAN data.

REQUIREMENTS: 
    
    - Libraries listed in Dependencies
    - Kaggle account and opendatasets installed if GTZAN dataset is to be 
    downloaded from there.

@author: NK
"""
#%% Dependencies
import pandas as pd
import numpy as np
import os
import librosa
import opendatasets as od # Only if downloading through Kaggle
#%%
def initialize_dataframe():
  '''
  Initializes an empty dataframe with columns for the features to be extracted,
  plus the filenames and genre labels
  '''
  # Features to be inlcuded:
  columns = ['filename', 
           'tempo',
           'harmonic_mean',
           'harmonic_var',
           'percussive_mean',
           'percussive_var',
           'chroma_stft_mean',
           'chroma_stft_var',
           'spectral_centroid_mean',
           'spectral_centroid_var',
           'zero_crossing_rate_mean',
           'zero_crossing_rate_var'
           ]
  # MFCCs from 1 to 20
  columns.extend([f'mfcc_{i+1}_mean' for i in range(20)])
  columns.extend([f'mfcc_{i+1}_var' for i in range(20)])
  # Add labels (genre) column
  columns.extend(['genre'])
  # Generate the empty dataframe
  dataframe = pd.DataFrame(columns = columns)

  return dataframe
#%%
def extract_features(root, file, feat_cols):
  '''
  Extracts all features from an audio file and stores them in a dataframe to be
  used as a single row.
  args:
      root : directory path of audio dataset
      file : filename of song as it appears in dataset
      feat_cols : column list for the feature dataframe
  returns:
      temp_df : a single row dataframe with extracted features for the specific
      song
      
  '''
  # Load file (sr = 22050 by default)
  y, sr = librosa.load(root+'/'+file)
  # Get filename
  filename = file
  # Calculate tempo
  tempo = librosa.beat.tempo(y)[0]
  # Calculate harmonic and percussive components
  harmonic, percussive = librosa.effects.hpss(y)
  # Calculate chroma STFT
  chroma_stft = librosa.feature.chroma_stft(y)
  # Calculate spectral centroid
  spectral_centroid = librosa.feature.spectral_centroid(y)[0]
  # Calculate zero crossing rate
  zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
  # Make features list up til now
  features = [filename,
              tempo,
              np.mean(harmonic),
              np.var(harmonic),
              np.mean(percussive),
              np.var(percussive),
              np.mean(chroma_stft),
              np.var(chroma_stft),
              np.mean(spectral_centroid),
              np.var(spectral_centroid),
              np.mean(zero_crossing_rate),
              np.var(zero_crossing_rate)
              ]

  # Calculate the MFCCs
  mfcc = librosa.feature.mfcc(y, sr, n_mfcc=20)

  mfcc_mean = []
  mfcc_var = []
  # Get list of mean and var for each component:
  for n in range(20):
    mfcc_mean.append(np.mean(mfcc[n,:]))
    mfcc_var.append(np.var(mfcc[n,:]))

  # Add the MFCCs to the feature list
  features.extend(mfcc_mean)
  features.extend(mfcc_var)
  
  # Add the genre to the list, taken from the filename
  genre = filename.partition('.')[0]
  features.extend([genre])

  # Store everything in dataframe format
  temp_df = pd.DataFrame(columns = feat_cols, data = [features])

  return temp_df
#%%
def GTZAN_feature_dataframe(audio_path):
  '''
  Creates a dataframe for all the songs in the dataset, and populates it with
  by extracting audio features for each. 
  args:
      audio_path : path to audio dataset directory
  returns:
      df : features dataframe for all songs in dataset
  '''
  # Initialize dataframe
  df = initialize_dataframe()

  # Go through every audio file in the dataset and extract features
  for root, dirs, files in os.walk(audio_path, topdown=False):
    for name in files:
        
      # Create a temporary single row dataframe of features
      temp_df = extract_features(root, name, df.columns)
      # Add new entry as a row to the main dataframe
      df.append(temp_df, ignore_index = True)

  return df
#%%
def GTZAN_name_dataframe(GTZANindex, feature_dataframe):
    '''
    Reads in the Artists and Song Titles list assembled by BL Sturm,
    associates filenames with the song names.
    args:
        GTZANindex : path to the artists and titles list file
        feature_dataframe : dataframe of features for all songs in dataset
    returns: 
        df_names : dataframe with artist name and song title corresponding
        to every filename in the dataset
    '''
    # read in names list as a dataframe:
    df_names = pd.read_table(GTZANindex, sep=' :::', header=None, engine='python')
    # set columns 
    df_names.columns = ['filename','Artist','Title']
    # fill missing values with N/A
    df_names.fillna('N/A', inplace= True)
    # get genres from features dataframe
    df_names = pd.merge(df_names,
                        feature_dataframe.filter(['filename','genre']),
                        how = 'inner',
                        left_on='filename',
                        right_on='filename')
    return df_names
#%%
def main():
    '''
    Runs pipeline to extract audio features from the GTZAN audio data and 
    generate a pandas dataframe.
    '''
    # Download dataset from Kaggle - !Requires account!
    # !Might need to pip install opendatasets!
    od.download('https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification')
    
    # Specify path to the audio data !MIGHT DIFFER!
    AUDIO_PATH = '../gtzan-dataset-music-genre-classification/Data/genres_original'    
    # Path to Artist/Title index:
    INDEX_PATH = './dataframes/GTZANindex.txt'
    
    # Generate feature dataframe
    df = GTZAN_feature_dataframe(AUDIO_PATH)
    # Generate names dataframe
    df_names = GTZAN_name_dataframe(INDEX_PATH)
    
    # Save as .csv
    df_path = './dataframes/feature_dataframe.csv'
    df.to_csv(df_path, index=False) 
    
    df_names_path = './dataframes/names_dataframe.csv'
    df_names.to_csv(df_names_path, index=False) 

#%%
if __name__ == "__main__":
    main()
