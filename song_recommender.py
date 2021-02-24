#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Program for getting similar recommendations on a song, with selections from the GTZAN dataset.

@author: nk
"""
#%% Dependencies:

import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import cosine_similarity

#%%
fdf = './dataframes/feature_dataframe.csv'
ndf = './dataframes/names_dataframe.csv'
# Choose song to get recommendations on (keep in mind the system will randomly consider a 30 excerpt)
SELECTED_SONG = 'path/to/song/you/like.wav'

#%%
class _Song_Recommender():
    '''
    Singleton Song Recommender Class, extracts features from an input song 
    (.wav format) and recommends a number of similar sounding songs from the 
    GTZAN dataset.
    
    initial args:
        feature_dataframe : dataframe with stored features
        names_dataframe : helper dataframe associating filenames and artist
        names/song titles
    '''
    feature_dataframe = pd.read_csv(fdf)
    names_dataframe = pd.read_csv(ndf)
        
    SAMPLE_RATE = 22050
    
    _instance = None
    
    def extract_features(self, filename, y, feat_cols):
        '''
        Extracts all features from an audio file and stores them in a dataframe to be
        used as a single row.
        '''
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
        mfcc = librosa.feature.mfcc(y, n_mfcc=20)
        
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
    
    def preprocess(self, song_name):
        
        data, _ = librosa.load(song_name)
        # Set length of segment to consider (30s long)
        duration = self.SAMPLE_RATE * 30
        
        # Pad to appropriate length...
        if len(data) < duration:
            max_offset = np.abs(len(data) - duration)
            offset = np.random.randint(max_offset)
            data = np.pad(data, (offset, duration-len(data)-offset), "constant")
        
        # ... or cut to appropriate length...
        elif len(data) > duration:
            max_offset = np.abs(len(data) - duration)
            offset = np.random.randint(max_offset)
            data = data[offset:len(data)-max_offset+offset]
        
        # ... or leave as is.
        else:
            offset = 0
        
        # Temporary row with extracted features:
        temp_df = self.extract_features(song_name, data, self.feature_dataframe.columns)
        # Add to the GTZAN feature dataframe:
        self.feature_dataframe = self.feature_dataframe.append(temp_df, ignore_index = True)
        
    def construct_similarity_matrix(self):
        '''
        Constructs similarity matrix
        '''
        #Extract labels
        df = self.feature_dataframe.set_index('filename')
        
        labels = df[['genre']]
        
        # Drop labels from original dataframe
        df = df.drop(columns=['genre', ])

        # Scale the data
        df = scale(df)
        
        # Cosine similarity
        similarity = cosine_similarity(df)

        # Convert into a dataframe and then set the row index and column names as labels
        sim_df_labels = pd.DataFrame(similarity)
        similarity_matrix = sim_df_labels.set_index(labels.index)
        similarity_matrix.columns = labels.index
        
        return similarity_matrix
    
    def recommend_songs(self, song_name, n_similar_songs):
        '''
        Finds similar songs based on cosine similarity between features
        '''
        self.preprocess(song_name)
        
        similarity_matrix = self.construct_similarity_matrix()
        
        series = similarity_matrix[song_name].sort_values(ascending = False)
        
        # Find most simlilar songs to selected:
        series = similarity_matrix[song_name].sort_values(ascending = False)
        
        print('\n You might enjoy the following songs from the GTZAN database:\n')
        # Initialize filenames list:
        similar_songs = []

        # Get artists and titles of the most similar songs:
        for i in range(1,n_similar_songs+1):
            # Locate by filename in GTZAN:
            fname = series.index[i]
            # Add to filenames list:
            similar_songs.append(fname)

            # Temporary row to store names:
            temp_row = self.names_dataframe.loc[
                    self.names_dataframe['filename'] == fname, 
                    ['Artist', 'Title', 'genre']]
            # Display Artist and Title
            print(str(temp_row.Artist.values[0])+
                  ' -'+str(temp_row.Title.values[0])+
                  ' ('+str(temp_row.genre.values[0])+')')

        return similar_songs
#%%
def Song_Recommender():
    '''
    Factory function for _Song_Recommender class.
    returns:
        _Song_Recommender._instance (_Song_Recommender)
    '''
    
    # Ensure an instance is only created the first time the factory function is
    # called:
    if _Song_Recommender._instance is None:
        _Song_Recommender._instance = _Song_Recommender()
    return _Song_Recommender._instance
#&&
if __name__ == "__main__":
    
    # create 2 instances:
    SR = Song_Recommender()
    SR0 = Song_Recommender()
    
    # check that different instances point back to the same object (Singleton):
    assert SR is SR0
    
    # make a recommendation
    recommended_songs = SR.recommend_songs(SELECTED_SONG, 5)
