#Python function to save the data
# It takes as input the json outputs of Alphapose and converts them to the appropriate dictionary form for
# the SLT methods
# Developed in the context of the aiD project
import gzip
import pickle
import torch
import json

def save(Data,name):

  with gzip.open(name, 'wb') as handle:
    pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def convert(json_path):
    '''
    Given a json file, convert it to a dictionary with the necessary keys for training the SLT model. 
    '''

    datas = []
    outputpath = json_path.replace('.json', '.test')
    with open(json_path, 'r') as f:
        data = json.load(f)

        
    data_tensor = torch.Tensor([ x['keypoints'] for x in data[3:]])

    if len(data_tensor) == 0:
        print('Did not extract any keypoints.. Please recheck the video..')
        return None
    
    dict_data = {
        'name': 'UploadedVideo',
        'signer': data[1]['signer'],
        'gloss': data[2]['gloss'],
        'text': data[0]['text'],
        'sign': data_tensor
        }

    datas.append(dict_data)
    save(datas, outputpath)
    
    return outputpath
