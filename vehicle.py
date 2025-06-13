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
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        for result in executor.map(detection_async, _requests):
            results.append(result)
            
    return results


if __name__ == "__main__":
    Dataset = "vowel"
    #time.sleep(10800)
    """requests = [{"trial" : f"{i}_{j}",
                "dataset" : Dataset, 
                "jarFile" : "target\MoFGBML-23.0.0-SNAPSHOT-Basic.jar", 
                "algroithmID" : f"Basic\{Dataset}_Basic",
                "parallelCores" : "6",
                "experimentID" : f"trial{i}{j}",
                "trainFile" : f"dataset\\{Dataset}\\a{i}_{j}_{Dataset}-10tra.dat",
                "testFile" : f"dataset\\{Dataset}\\a{i}_{j}_{Dataset}-10tst.dat"} \
                for i in range(3) for j in range(10)]"""
    
    start_i = 1 # Start from i = 1
    start_j = 8 # Start from j = 8 when i = 1
    requests = []
    for i in range(3):
        for j in range(10):
            # Skip trials that are before our desired starting point
            if (i < start_i) or (i == start_i and j < start_j):
                continue
            
            requests.append({
                "trial": f"{i}_{j}",
                "dataset": Dataset,
                "jarFile": r"target\MoFGBML-23.0.0-SNAPSHOT-Basic.jar", # Use raw string for backslashes
                "algroithmID": f"Basic\{Dataset}_Basic",
                "parallelCores": "6",
                "experimentID": f"trial{i}{j}",
                "trainFile": f"dataset\\{Dataset}\\a{i}_{j}_{Dataset}-10tra.dat",
                "testFile": f"dataset\\{Dataset}\\a{i}_{j}_{Dataset}-10tst.dat"
            })
    
    for result in detection_async_parallel(requests):
        
        print(result)