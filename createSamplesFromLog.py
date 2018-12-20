def createSamplesFromLog():
    samples = []
    with open('data/driving_log.csv') as csvfile:
        import csv
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
        samples = samples[1:] # throw away header line

    from sklearn.model_selection import train_test_split
    return train_test_split(samples, test_size=0.2, shuffle=True)