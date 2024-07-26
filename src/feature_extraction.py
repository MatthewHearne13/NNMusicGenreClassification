import librosa
import os
import csv
import pandas

def FeatureExtraction():
    """Creates a CSV file saved at ___ called featureExtracted.csv"""
    # first we create a row header for the csv
    header = 'filename'
    for i in range(1, 21):
        header += f' mfcc{i}'
    for j in range(1, 16):
        header += f' lpc{j}'
    header += f' zero_crossing_rate rolloff spectral_centroid'

    # formatting for the csv
    header = header.split()

    # create the file and add the header
    file = open('featureExtracted.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    musicGenres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

    # loop through all the music files in the folder music
    # load the files with librosa and extract mfcc and lpc features
    # then add them to the csv
    for genre in musicGenres:

        for filename in os.listdir(f'./'):
            songname = filename
            y, sr = librosa.load(songname, mono=True, duration=30) # change the duration according to our test
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            lpc = librosa.lpc(y=y, order=16)  # order should be between 10 and 20 and 16 usually used
            row = f'{filename}'
            for m in mfcc:
                row += f' {np.mean(m)}'
            for l in lpc:
                row += f' {np.mean(l)}'
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            rolloff = librosa.feature.spectral_rolloff( y=y, sr=sr)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            row += f' {np.mean(zero_crossing_rate)} {np.mean(rolloff)} {np.mean(spectral_centroid)}'
            file = open('featureExtracted.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(row.split())

def PreProcess():
    """Preproccesses the data from the featureExtracted.csv file"""
    data = pandas.read_csv('featureExtracted.csv')
    data.head() # Maybe don't need
    data = data.drop(['filename'], axis=1)
    data.head() # Again maybe don't need

    # Create a mapping where each genre is represented by an integer
    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)
