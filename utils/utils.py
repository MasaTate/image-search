import random
from PIL import Image
from PIL.ExifTags import TAGS
import datetime


def randomIntList(length:int, min:int, max:int):
    out = []
    
    assert length <= max - min + 1
    
    while(len(out) < length):
        n = random.randint(min, max)
        if not n in out: out.append(n)
        
    return out

def getDate(path:str):
    im = Image.open(path)

    try:
        exif = im._getexif()
    except:
        return None

    exif_table={}
    for tag_id, value in exif.items():
        tag = TAGS.get(tag_id, None)
        exif_table[tag] = value

    return exif_table['DateTimeOriginal']

def strToDate(date: str):
    if ' ' in date:
        date = date.split(' ')[0]

    return datetime.datetime.strptime(date, '%Y:%m:%d').strftime("%Y-%m-%d")

