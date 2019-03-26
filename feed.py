from os import fsencode, listdir, fsdecode
import queue
import numpy as np

def Batch(data):
    batches = []
    tempBatch = []
    max_size = 40
    for x in range(data.qsize()):
        tempData = data.get()
        #print(tempData)
        if tempData == [-1, -1]:
            tempBatch.append(tempData)
            batches.append(tempBatch)
            tempBatch = []
            continue
        if len(tempBatch) < max_size:
            tempBatch.append(tempData)
        elif len(tempBatch) >= max_size:
            tempBatch.append(tempData)
            batches.append(tempBatch)
            tempBatch = []
        else:
            print("USELESS ERROR MESSAGES!")
    #print(batches)
    return batches

def Scavenge():
    data = queue.Queue()
    directory = fsencode("D:/MMP/TechnicalWork/Data")
    for file in listdir(directory):
        filename = fsdecode(file)
        if filename.endswith(".csv"):
            with open("Data/" + filename, "r") as music_file:
                tempDataNotes = []
                for line in music_file:
                    temp = line.split(", ")
                    tempData = []
                    if line != '\n':
                        if temp[2] == "Note_on_c":
                            tempData.append(int(temp[1]))
                            tempData.append(int(temp[4]))
                            tempDataNotes.append(tempData)
                        elif temp[2] == "Note_off_c":
                            tempData.append(int(temp[1]))
                            tempData.append(int(temp[4]))
                            tempDataNotes.append(tempData)
                        elif (temp[2] == "End_of_file\n" or temp[2] == "End_track\n") and temp[0] != str(1):
                            data.put([-1,-1])
                    if(len(tempDataNotes) >= 2):
                        length_note_combo = tempDataNotes[1][0] - tempDataNotes[0][0], tempDataNotes[0][1]
                        data.put(length_note_combo)
                        tempDataNotes = []
    tempBatch = Batch(data)
    finalBatch = [batch for batch in tempBatch if len(batch) >= 41]
    #for batch in finalBatch:
    #    print(len(batch))
    return finalBatch

# Comment when live, uncomment for testing
#Scavenge()