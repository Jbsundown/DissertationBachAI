import py_midicsv
import numpy as np
import random
from csv import writer, reader
from os import system, listdir


def Time(filename):
  temp_time_holder = 0
  time = 0
  iteration = 0
  with open(filename, 'r') as file:
      for line in file:
          splits = line.split(",")
          temp_time_holder = splits[0]
          with open(filename[:-4] + "_converted.csv", 'a') as output_file:
              output_file.write("2," + str(time) + ",Note_on_c," + "0," + splits[1].strip("\n") + ",64\n")
              time+=int(float(temp_time_holder))
              output_file.write("2," + str(time) + ",Note_off_c," + "0," + splits[1].strip("\n") + ",0\n")
              if(iteration >= 49):
                  output_file.write("2,1000000,End_track\n")
                  output_file.write("0,0,End_of_file\n")
              iteration+=1

def Build(filename):
  with open(filename[:-4] + "_converted.csv", "a+") as output_file:
     output_file.write("0,0,Header,1,2,96\n")
     output_file.write("1,0,Start_track\n")
     output_file.write("1,0,Key_signature,-1,major\n")
     output_file.write("1,0,Time_signature,8,2,24,8\n")
     output_file.write("1,0,Tempo,599704\n")
     output_file.write("1,10000,End_track\n")
     output_file.write("2,0,Start_track\n")
     output_file.write("2,0,Program_c,0,0\n")
  Time(filename)
        

def Convert(filename):
  system("D:/mmptestenv/outputs/Csvmidi.exe " + filename + " " + filename[:-4] + ".mid")


def Work(filename):
  Build(filename)
  Convert(filename[:-4] + "_converted.csv")

def Extract(filename):
  #Used to build MIDI extracts from genuine Bach pieces
  with open(filename, 'r') as file:
    read = reader(file)
    file_data = list(read)
  if(len(file_data) < 80):
    return
  lines = len(file_data)
  start_point = np.random.randint(lines)
  extracted_music = []
  while(start_point > lines-82):
    #If the number of lines will overflow the length of the file
    start_point = np.random.randint(lines)
  if (start_point %2 == 0):
    #If starting on a note_off, start on a note_on.
    start_point+=1
  time_value = int(file_data[start_point][1])
  for line in range(start_point, start_point+81):
    note_data = []
    note_data.append(2)
    note_data.append(int(file_data[line][1])-time_value)
    note_data.append(file_data[line][2])
    note_data.append(0)
    note_data.append(file_data[line][4])
    note_data.append(64 if int(file_data[line][5]) > 0 else 0)
    extracted_music.append(note_data)
  with open("Highlights/Test/" + filename[5:-4]+'extracted.csv', 'a+', newline='') as wrtitingFile:
    wrtitingFile.write("0,0,Header,1,2,96\n")
    wrtitingFile.write("1,0,Start_track\n")
    wrtitingFile.write("1,0,Key_signature,-1,major\n")
    wrtitingFile.write("1,0,Time_signature,8,2,24,8\n")
    wrtitingFile.write("1,0,Tempo,599704\n")
    wrtitingFile.write("1,10000,End_track\n")
    wrtitingFile.write("2,0,Start_track\n")
    wrtitingFile.write("2,0,Program_c,0,0\n")
    write = writer(wrtitingFile)
    write.writerows(extracted_music)
    wrtitingFile.write("2,1000000,End_track\n")
    wrtitingFile.write("0,0,End_of_file\n")
  Convert("Highlights/Test/" + filename[5:-4]+'extracted.csv')
    
#Extracts random pieces of Bach music and converts them into MIDI matching the same format as the generated music
# for x in range(0,20):
#   chosen = random.choice(listdir("D:/MMP/TechnicalWork/Data/"))
#   print(chosen)
#   Extract("Data/{}".format(chosen))

