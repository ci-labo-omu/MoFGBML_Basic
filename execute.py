# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 01:04:14 2022

@author: kawano
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor
import time

def detection_async(_request):
    
    result = subprocess.run(["Java", "-Xms1g", "-Xmx8g", "-jar",
                             _request["jarFile"],
                             _request["dataset"],
                             _request["algroithmID"],
                             _request["experimentID"],
                             _request["parallelCores"],
                             _request["trainFile"],
                             _request["testFile"]])
    
    return result


def detection_async_parallel(_requests):
    
    """results = []
    
    for req in _requests:
        result = detection_async(req)
        results.append(result)
            
    return results"""
    results = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        for result in executor.map(detection_async, _requests):
            results.append(result)
            
    return results


if __name__ == "__main__":
    Dataset = "vehicle"
    requests = [{"trial" : f"{i}_{j}",
                "dataset" : Dataset ,
                
                
                "jarFile" : "target\MoFGBML-23.0.0-SNAPSHOT-Basic.jar", 
                "algroithmID" : f"Basic\{Dataset}_Basic",
                "parallelCores" : "6",
                "experimentID" : f"trial{i}{j}_art10",
                "trainFile" : f"dataset_nodes\\{Dataset}\\minCIM_10\\a{i}_{j}_{Dataset}_nodes_ART.csv",
                "testFile" : f"dataset\\{Dataset}\\a{i}_{j}_{Dataset}-10tst.dat"} \
                for i in range(3) for j in range(10)]


    
    for result in detection_async_parallel(requests):
        
        print(result)