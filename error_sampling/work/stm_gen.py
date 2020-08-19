"""
This script takes in reference mlf and a name map file
Converts the mlf into an .stm file
"""
import sys, os

inmlf = sys.argv[1]
mapfile = sys.argv[2]
outstm = inmlf[:-4] + '.stm'
namemap = {}

with open(mapfile) as fin:
    for line in fin:
        lineelems = line.split()
        _, meeting_name, headset, speaker = lineelems[2].split('_')
        headset_name = meeting_name + '.Headset-' + headset[-1]
        namemap[lineelems[0]] = [headset_name, '0'] + lineelems[2:]

to_write = []
current_utt = []
with open(inmlf) as fin:
    for line in fin:
        if line[0] == "\"":
            refname = line.strip().strip("\"")[:-4]
            current_utt += namemap[refname]
        elif line[0] == '.':
            to_write.append(' '.join(current_utt) + '\n')
            current_utt = []
        elif line[0] != '#':
            current_utt.append(line.strip())

with open(outstm, 'w') as fout:
    fout.writelines(to_write)
