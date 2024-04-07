import subprocess
from PIL import Image
from PIL.ExifTags import TAGS

imgPath = '/home/GTL/tsutar/Traffic_sign_detection/datasets/LISA_yolo/vid_frames/addedLane_1323813414-avi_image0_png_jpg.rf.8edfd337fba6341a647f39ffba0bdae4.jpg'
exeProcess = "hachoir-metadata"
process = subprocess.Popen([exeProcess,imgPath],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True)

infoDict = {}  # Initialize the infoDict dictionary

# for tag in process.stdout:
#     line = tag.strip().split(':')
#     infoDict[line[0].strip()] = line[-1].strip()

# for k,v in infoDict.items():
#     print(k,':', v)

image = Image.open(imgPath)
exifdata = image.getexif()

# iterating over all EXIF data fields
for tag_id in exifdata:
    # get the tag name, instead of human unreadable tag id
    tag = TAGS.get(tag_id, tag_id)
    data = exifdata.get(tag_id).decode("utf-16")
    print(f"{tag:25}: {data}")  
