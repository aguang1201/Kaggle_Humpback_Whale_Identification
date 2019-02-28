import numpy as np
import glob


fknown_all = np.zeros(shape=(15696, 512), dtype=np.float32)
fsubmit_all = np.zeros(shape=(7960, 512), dtype=np.float32)
for file_fknown in glob.glob("TTA_file/fknown*.npy"):
    fknown_array = np.load(file_fknown)
    fknown_all = fknown_all + fknown_array

for file_fsubmit in glob.glob("TTA_file/fsubmit*.npy"):
    fsubmit_array = np.load(file_fsubmit)
    fsubmit_all = fsubmit_all + fsubmit_array

file_len = len(glob.glob("TTA_file/fknown*.npy"))
mean_fknown = fknown_all / file_len
mean_fsubmit = fsubmit_all / file_len

np.save('TTA_file/mean_fknown.npy', mean_fknown)
np.save('TTA_file/mean_fsubmit.npy', mean_fsubmit)