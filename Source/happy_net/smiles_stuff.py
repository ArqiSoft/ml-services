lines = open('smile_line.csv', "r").readlines()
out = open('smiles_20.smi', "w")
d = dict((key, 0) for key in range(0,120))
print(d)

for line in lines:
    if len(line.strip()) == 20:
        out.write(line)