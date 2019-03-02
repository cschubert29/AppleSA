import re
import json


#STEP 1: READ FILE AND DELETE BLANK LINES

def deleteBlanks():
    newFile = open('NEW_FILE_NAME.json', 'w')
    oldFile = "OLD_FILE_NAME.json"
    with open(oldFile, 'r') as f:
        print ("".join(line for line in f if not line.isspace()), file=newFile)