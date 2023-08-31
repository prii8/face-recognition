import json

dictobj={}
#with open("data.json") as fp:
    #dictobj=json.load(fp)
#print(dictobj)
f = open("data.json","r")
dictobj=json.load(f)
print(dictobj)
id = "Name0"
if id not in dictobj.keys():
    dictobj[id] = "Person_3"
    print(dictobj)
    abc = json.dumps(dictobj)
    f = open("data.json", "w")
    f.write(abc)
    f.close()
else:
    print("id already exsits")

