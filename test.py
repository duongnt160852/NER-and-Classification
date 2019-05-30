import json
f = open('data/giaitri.json','r',encoding = 'utf-8-sig') #mở file
for line in f:
    data=json.loads(line)
    print(str(data['title']).strip())
f.close()



