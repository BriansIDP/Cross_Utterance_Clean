import sys, os

countfile = sys.argv[1]
count = 0

with open(countfile) as fin:
    for line in fin:
        words = line.strip().split()
        count += 1
        for word in words:
            count += 1
print(count)
