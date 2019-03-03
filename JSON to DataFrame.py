

'''
Florida International Univeristy - Data Science MS
CAP 5640 - NLP - Spring 2019
Andrea Garcia and Constanza Schubert

JSON files to Python Dataframe
'''

# STEP 1: READ FILE AND DELETE BLANK LINES

def deleteblanks(filetoedit, newfilename):
    #Accepts two .json files as input, original file name (needs cleaning) and new file name (function will create new  .json clean file under this name) 
    newfile = open(newfilename, 'w')
    with open(filetoedit, 'r') as f:
        print ("".join(fileline for fileline in f if not fileline.isspace()), file=newfile)

    return

# STEP 2: CREATE NESTED DICTIONARY

def createdict(filename):
    #Accepts a .json file as input and converts data into a nested dictionary. File must not contain any empty lines before, throughout or after main body.
    with open(filename, "r") as jsondata:
        tweetlines = []
        for tline in jsondata:
            tweetlines.append(tline)

    tweetdict = {}
    i = 0
    while i < len(tweetlines):
        tweetdict[i + 1] = json.loads(tweetlines[i])
        i += 1
    return tweetdict


# STEP 5: CREATE DATAFRAME FROM DICTIONARY

def createdtframe(tweetdict):
    #Accepts python dictionary as input and converts to dataframe using keys as columns and values as rows
    #Columns comprised of only top level keys
     df = pd.DataFrame.from_dict(tweetdict, orient='index')
    return df