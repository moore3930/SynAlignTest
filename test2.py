import tensorflow as tf

# Function in python
path = './data/en-de.txt'
path_out = './data/en-de-format.txt'
fout = open(path_out, 'w')
fin = open(path)
for line in fin:
    line_array = line.strip().split('\t')
    if len(line_array) == 2:
        fout.write(line)
fout.flush()
fout.close()
fin.close()

