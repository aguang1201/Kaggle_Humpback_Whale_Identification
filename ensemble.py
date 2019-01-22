import csv
import pandas as pd # not key to functionality of kernel
from datetime import datetime

# sub_files = [
#              '../input/weighted-average-ensemble-v-n-lb-0-776/sub_ns.csv',
#              '../input/ensembling-voting-lb-0-777/sub_ens.csv',
#              '../input/ensembling-voting-lb-0-777/sub_ens.csv'
#             ]
sub_files = [
             'submit/sub_ens_2019-01-02.csv',
             'submit/submission_0822.csv'
            ]

# Weights of the individual subs
sub_weight = [
              0.777**2,
              0.822**2
             ]

abc = pd.read_csv(sub_files[0])
xyz = pd.read_csv(sub_files[1])

print(abc.head())
print(xyz.head())

Hlabel = 'Image'
Htarget = 'Id'
npt = 5  # number of places in target

place_weights = {}
for i in range(npt):
    place_weights[i] = (1 / (i + 1))

print(place_weights)

lg = len(sub_files)
sub = [None] * lg
for i, file in enumerate(sub_files):
    ## input files ##
    print("Reading {}: w={} - {}".format(i, sub_weight[i], file))
    reader = csv.DictReader(open(file, "r"))
    sub[i] = sorted(reader, key=lambda d: str(d[Hlabel]))

## output file ##
time_now = datetime.now()
out = open(f"submit/sub_ens_{time_now}.csv", "w", newline='')
writer = csv.writer(out)
writer.writerow([Hlabel, Htarget])

for p, row in enumerate(sub[0]):
    target_weight = {}
    for s in range(lg):
        row1 = sub[s][p]
        for ind, trgt in enumerate(row1[Htarget].split(' ')):
            target_weight[trgt] = target_weight.get(trgt, 0) + (place_weights[ind] * sub_weight[s])
    tops_trgt = sorted(target_weight, key=target_weight.get, reverse=True)[:npt]
    writer.writerow([row1[Hlabel], " ".join(tops_trgt)])
out.close()