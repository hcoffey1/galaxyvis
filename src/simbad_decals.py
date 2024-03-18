#Take in decals data and perform lookup in simbad
import time
import pandas as pd
import requests

file="./data/gz_decals_auto_posteriors.parquet"

data = pd.read_parquet(file)


def create_url(ra, dec, radius):
    url = f'http://simbad.u-strasbg.fr/simbad/sim-coo\
?Coord={ra}d{dec}d\
&CooFrame=ICRS&CooEpoch=2000\
&CooEqui=2000\
&Radius={radius}\
&Radius.unit=arcmin\
&submit=submit+query\
&CoordList=&output.format=ASCII'

    return url

def pull_object(ra, dec):
    minRadius = 0.1
    radiusInc = 0.1
    maxRadius = 1.5 
    radius = minRadius 

    maxRequests = 15 
    rCount = 0

    print("ra :", ra)
    print("dec :", dec)
    while radius <= maxRadius and rCount < maxRequests:
        url = create_url(ra,dec,radius)
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            print("requests:Encountered timeout, waiting, then restarting.")
            time.sleep(30)
            print("Continuing")
            continue

        if response.status_code == 200:
            text_content = response.text
            if "No astronomical object found" in text_content:
                print("No object found at radius", radius, "increasing search range.")
                radius += radiusInc
            else:
                return text_content
        else:
            print("ERROR: ", response.status_code)
            return None 
        
        rCount += 1

    return None 

print(len(data['ra']))

for i in range(187678, len(data['ra'])):
    #print('--VVV-NEXT-ENTRY-VVV--')
    ra = data['ra'][i]
    dec = data['dec'][i]
    print("index :", i)

    with open('decals_scrape.txt', 'a') as f:

    #url = create_url(ra,dec)
    #print("URL: ", url)

        obj = pull_object(ra,dec)
        f.write('GalaxyVis:ENTRY\n')
        f.write("index : " + str(i) + '\n')
        f.write("ra : " + str(ra) + '\n')
        f.write("dec : " + str(dec) + '\n')
        if obj != None:
            f.write(obj)
        else:
            f.write('GalaxyVis:ERROR\n')
    #break

    #lbreak

    #response = requests.get(url)

    #if response.status_code == 200:
    #    text_content = response.text
    #    print(text_content)
    #else:
    #    print("ERROR: ", response.status_code)

    #print('dec,', data['dec'][i])
    #requests.get()
#for d in data:
#    print(d)
