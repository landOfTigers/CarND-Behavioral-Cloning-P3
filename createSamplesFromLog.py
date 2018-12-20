import csv
from sklearn.model_selection import train_test_split
    
def readFromLogFile(fileName):
    samples = []
    with open(fileName) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
        samples = samples[1:] # throw away header line
    return samples

def createSamplesFromLog():
    samples = readFromLogFile('data/driving_log.csv')
    samples.extend(readFromLogFile('data/driving_log_backwards.csv'))
    return train_test_split(samples, test_size=0.2, shuffle=True)