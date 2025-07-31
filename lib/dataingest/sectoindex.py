import json

file = open("data\\cik\\raw_cikmap.json", "r")
data = json.load(file)
file.close()

res = {data[i]["ticker"] : {"cik" : data[i]["cik_str"], "name" : data[i]["title"]} for i in data}

file = open("data\\cik\\cikmap.json", "w")
json.dump(res, file)
file.close()