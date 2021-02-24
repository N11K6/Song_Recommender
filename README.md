# Content-Based Song Recommendations from the GTZAN Dataset

This is a song recommender service program based on the GTZAN song database. Its aim is to find similar sounding songs from the available pool, using purely content-based criteria. It avoids subjective, arbitrary, or nebulous labels such as assigned genre, and aims to use similarity across features extracted through well-documented processes such as Chroma STFT, MFCCs etc. 

Recommended usage is to input an audio file (.wav), ideally a segment most characteristic of the overall tone of the song. The recommender algorithm then outputs a number of songs that exist in the database and sound most alike the chosen song.

## The Dataset

The GTZAN dataset was assembled within the MARSYAS framework (Music Analysis, Retrieval and Synthesis for Audio Signals) helmed by George Tzanetakis (http://marsyas.info/index.html) [1]. The dataset consists of 30s long segments from various songs, arranged by their assigned genre. 100 songs are included for each of 10 genres to make a total of 1000 in the dataset. Since its release, it has been a popular choice for subsequent work on musical genre classification using machine learning.

Despite its popularity and useability, there are some practical issues when using the raw dataset. Perhaps most prevalent in the scope of this project, is the absence of Artist Names or Song Titles associated with each audio file. This has been pointed out by B. L. Sturm [2], who also took upon the task to assemble a list of all names and titles that could be traced. This list is used in the project.

## Approach

As mentioned, the aim of this project is to offer a song recommendation service, therefore it will steer away from the usual task of genre classification that is associated with the GTZAN dataset. The recommendation algorithm is to be entirely content-based, meaning that it only considers features that are directly extracted from the audio instead of assigned labels. 

1. For each audio file in the dataset, a number of features are extracted from its waveform. In order to get a single numerical value for each feature, many of them have to be averaged over the duration of the sample. To account for this averaging, the mean value and variance of such features are kept.
2. The assembled dataframe contains 1000 rows, one for each song, and columns corresponding to each feature.
3. From this feature dataframe, a 1000x1000 similarity matrix is assembled containing the distances of each song from all the others, using cosine similarity as a metric.
4. The most similar sounding songs relative to a chosen song can be located simply by taking the values corresponding to the highest similarity.
5. A helper dataframe, associating artist and song title to each filename in the dataset, is generated using the names list that is available. The filenames with highest similarity from the above matrix are used to identify each song.
6. The recommendations are provided by artist and song title. The genre is also included, to give an indication of the content for unidentified entries.

The above is the process for finding similar songs within the existing dataframe. A program is also designed to receive any audio file as input, and reommend songs from the dataset depending on the similarity of their content. The pipeline is as follows:

1. The features of the input audio file are extracted and added as an extra row to the main features dataframe.
2. The similarity matrix is recalculated to indlude the new entry.
3. Most similar songs to the new entry are selected following the process above.
4. The recommender prints out a list of similar songs by their artist, title and genre.

## Results

An example of finding similar songs based on an existing entry is as follows:

> If you like

>   Sting - Children's Crusade

> 

>  You might enjoy the following songs:

> 

>  Sting - Consider Me Gone (rock)

>  Kate Bush - Couldbusting (pop)

>  Sting - Russians (rock)

>  Nina Martinique - Silver Threads And Golden Needles (country)

>  Desmond Dekker - Shanty Town (reggae)


It can be noted that assigned genre appears to bear no correlation to the results, however, these songs can be found to still sound similar.

Using a new song as an input (Sebastian - Love in Motion), the following recommendations are output:

> You might enjoy the following songs from the GTZAN database:

>  The Tymes - You Little Trustmaker (disco)

>  nan -nan (jazz)

>  Public Enemy - Reggie Jax (hiphop)

>  Carl Carlton - She's A Bad Mama Jama (disco)

>  Shania Twain - You Win My Love (country)


Again the algorithm has chosen songs based on similar sound. Note that the jazz recommendation is unidentified, but at least the genre exists to give an slight indication.

## Acknowledgements

Credit should be given to Andrada Olteanu, who uploaded the dataset in this form on Kaggle(https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification), and has provided well documented examples features extraction, classification tasks, and a basis for a recommender system upon which this work expands.

## References

> [1] Tzanetakis, George & Cook, Perry. (2002). Musical Genre Classification of Audio Signals. IEEE Transactions on Speech and Audio Processing. 10. 293 - 302. 10.1109/TSA.2002.800560. 

> [2] Sturm, B. L. (2013). The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use. arXiv.org (e-prints), 1-29. http://arxiv.org/abs/1306.1461

