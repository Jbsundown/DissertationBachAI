import py_midicsv
from csv import writer, reader
from os import fsencode, listdir, fsdecode, remove

def Convert(filename):
    csv_output = py_midicsv.midi_to_csv(filename)
    with open(filename[:-4] +".csv", "w+") as output_file:
        midi_writer = writer(output_file, delimiter='\n')
        midi_writer.writerow(csv_output)

def Melodical(filename):
    #Checks to see if the notes are played one after the other in a melodical style
    mentioned = {}
    with open(filename, 'r') as file:
        next(file)
        for line in file:
            splits = line.split(", ")
            if splits[2][0] == "N":
                if not splits[1] in mentioned:
                    mentioned[splits[1]] = 1
                else: 
                    mentioned[splits[1]] += 1
                    if mentioned[splits[1]] >2:
                        print(filename, "IS NOT melodical as", splits[1], "appears too many times.")
                        return False
        print(filename, "is melodical")
        return True


def Split(filename):
    filename = filename[:-4] + ".csv"
    with open(filename, 'r') as file:
        current_track = 2
        for line in file:
            if line[0].isdigit() and int(line[0]) > 1 :
                if line[0] == current_track:
                    with open("Data/" + filename[:-4] + "track" + current_track + ".csv", 'a') as output_file:
                        output_file.write(line)
                    if line.split(", ")[2] == "End_track\n":
                        if not Melodical("Data/" + filename[:-4] + "track" + current_track + ".csv"):
                            remove("Data/" + filename[:-4] + "track" + current_track + ".csv")
                else:
                    current_track = line[0]
                    with open("Data/" + filename[:-4] + "track" + current_track + ".csv", 'w+') as output_file:
                        print("new file", filename[:-4] + "track" + current_track + ".csv", "created")
                        output_file.write(line)


directory = fsencode("D:/MMP/TechnicalWork")
for file in listdir(directory):
    filename = fsdecode(file)
    if filename.endswith(".mid"):
        print("Converting", filename)
        Convert(filename)
        print(filename, "converted")
        print("Splitting", filename[:-4] + ".csv")
        Split(filename)
        print(filename[:-4] + ".csv", "split")