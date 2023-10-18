import os
import subprocess
import time
from prepro import RunPrepro,RunPreproList


def Translate(output):
	bashCommand = "python3 -m signjoey test configs/PlayGround.yaml --output_path ../Results/{}".format( output)
	print(bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
	print(error)



Old_lines=[]
while True:

    with open('../TransOrders.txt', 'r') as file:
   
        lines = file.readlines()
        
    time.sleep(2)
    if lines!=Old_lines:
        dir_list=[]
        for line in lines:
           print(line)
           dir_list.append('../Trajectories/'+line[:-1]+'/'+line[:-1].split('.')[0])
        RunPreproList(dir_list,'../ToTranslate/TestData')
        Translate(line)
        open('../TransOrders.txt', 'w').close()
    Old_lines=lines
    



print(output)

