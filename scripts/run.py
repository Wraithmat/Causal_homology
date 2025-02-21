from pyclassify.classifier import kNN
from pyclassify.utils import read_file, read_config
import sys
import random

if __name__=='__main__':
    
    config_path = None
    for arg in sys.argv:
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
    dictionary = read_config(config_path)
    features, labels = read_file(dictionary['dataset'])
    
    temp = list(zip(features, labels))
    random.shuffle(temp)
    features, labels = zip(*temp)

    train_f = features[:int(0.8*len(labels))]
    train_l = labels[:int(0.8*len(labels))]

    test_f = features[int(0.8*len(labels)):]
    test_l = labels[int(0.8*len(labels)):]
     
    knn_classifier = kNN(k=dictionary['k'], backhand=dictionary.get('backhand','plain'))
    predictions = knn_classifier((train_f,train_l),test_f)
    

    tp=0
    for i in range(len(predictions)):
        if predictions[i]==test_l[i]:
            tp+=1

    print(tp/len(predictions))
