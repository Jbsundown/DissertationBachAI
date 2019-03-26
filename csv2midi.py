import py_midicsv
from csv import writer, reader
from os import fsencode, listdir, fsdecode, remove, system


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
                if(iteration >= 39):
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
    system("D:/MMP/TechnicalWork/outputs/Csvmidi.exe " + filename + " " + filename[:-4] + ".mid")


def Work(filename):
    Build(filename)
    Convert(filename[:-4] + "_converted.csv")
