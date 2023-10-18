# Python script to convert between data formats.
# It takes as input the json outputs of Alphapose and converts them to the appropriate dictionary form for
# the SLT methods. This script assumes that AlphaPose extraction has already been performed. 
# Developed in the context of the aiD project
import gzip
import pickle
import torch
import json

def save(Data,name):

  with gzip.open(name, 'wb') as handle:
    pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)

directory = 'data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test/'
text = 'data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv'
count = 0
all_data = []
with open(text, 'r') as csvfile:
    for line in csvfile:
        if count == 0:
            count +=1
            continue
        #print(line.split('|'))
        dir, _, _, _, signer,gloss, text = line.split('|')
        outputpath = directory + dir + '/'

        with open(outputpath + 'AlphaPose_out.json') as f:
            data = json.load(f)

        data_tensor = torch.Tensor([ x['keypoints'] for x in data[3:]])

        dict_data = {
            'name': dir,
            'signer': signer,
            'gloss': gloss.strip(),
            'text': text.strip(),
            'sign': data_tensor
            }

        all_data.append(dict_data)
        count += 1

        if count % 200 == 0:
            print(count)

save(all_data, 'alphapose_phoenix.test')
