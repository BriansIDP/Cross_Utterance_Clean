import sys, os

inputs = sys.argv[1]
outputs = inputs+'.mlf'

with open(inputs) as fin:
    lines_to_write = []
    for line in fin:
        elems = line.strip().split(' ')
        labname = elems[0]
        lines_to_write.append('"' + labname + '.lab"\n')
        for word in elems[1:]:
            if word[0] == '\'':
                word  = '\\' + word
            lines_to_write.append(word + '\n')
        lines_to_write.append('.\n')
with open(outputs, 'w') as fout:
    fout.writelines(lines_to_write)
