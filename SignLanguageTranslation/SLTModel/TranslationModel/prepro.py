import numpy as np
import pandas as pd
import json

import pickle
import pandas as pd
import gzip
import torch
from torch import tensor
import numpy as np
import sys

from copy import deepcopy
import sys


#point types
P=['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']

#non zero pose points
effective=np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 45, 46, 47, 48, 49, 50, 51,
            52, 53, 54, 55, 56])

#save  zip+pickle
def save(Data,name):
    
    with gzip.open(name, 'wb') as handle:
        pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Normalization over the time axs
def Norma(X):

    X=np.array(X)
    X=np.where(X==0,X.mean(axis=1)[0],X-X.mean(axis=1,keepdims=True))
    X=np.where(X==0,X.mean(axis=1)[0],X-X.mean(axis=1,keepdims=True))
    return X



#freame firectory
def getdir(video_dir,f=10):
    
    f=str(1000000000000+f)[1:]
    dir=video_dir+'_'+f+'_keypoints.json'
    
       
    return dir

#load frame traj
def get_frame(dir):
     
    
 
    # returns JSON object as
    # a dictionary
    
    f = open(dir)
    data = json.load(f)
    
    return data['people'][0]

#get effective xy point
def XY(data,key='face_keypoints_2d'):
    if key=='pose_keypoints_2d':
        
            x=np.array(data[key])[effective][::3]
            x=x/100
            y=np.array(data[key])[effective][1::3]
            y=y/100
            return x,y
    
    x=np.array(data[key][::3])
    x=x/100
    y=np.array(data[key][1::3])
    y=y/100
    
    return x,y

def mix(video_dir):
    X={p:[] for p in P}
    Y={p:[] for p in P}
    OUT=pd.DataFrame()
    started=False
    print(X)
    for i in range(1000):
        
        dir=getdir(video_dir,f=i)
        
        try:
           
           data= get_frame(dir)
           
           for p in P:
            x,y=XY(data,p)
            X[p].append(x)
            Y[p].append(y)
            started=True
          
        except Exception as e:
            
            if started:  #done
                
                break
                
            else:        #problimatic frame
                
                print('error frame ',i)
                pass

    for p in P:
        X[p]=Norma(X[p])
        Y[p]=Norma(Y[p])
    OUT=pd.concat([ pd.DataFrame(X[p]) for p in P ]+[ pd.DataFrame(Y[p]) for p in P ],axis=1)
    return OUT



def build_single(sign,name='Result'):
    
    try:
        out={}
        out['name']=name
        out['signer']='Unknown'
        out['gloss']='Unknown'
        out['text']='Unknown'
        out['sign']=torch.tensor(sign.values)
        return out
    except:
        print('can not build')
        return None
    
    
def build_all(df):
    out=[]
    for index, row in df.iterrows():
        unit=build_single(row )
        if unit != None:
            out.append(unit)
    return out





def RunPrepro(dir,name):
        sign=mix(dir)
        sign=[build_single(sign,name)]
        save(sign,name)

def RunPreproList(dir_list,name):
        if len(dir_list)==0:
              return True
        print(dir_list)
        
        
        sign=[build_single(mix(dir),name=dir)  for dir in dir_list]
        save(sign,name)

def main():
    
    dir = sys.argv[1]
    sign=mix(dir)
    sign=[build_single(sign)]
    save(sign,'TestData')
    

if __name__ == "__main__":
    main()
