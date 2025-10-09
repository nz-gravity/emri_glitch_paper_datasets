import numpy as np
import glob
import h5py
import matplotlib.pyplot as plt


max_glitch_SNR_list = [np.inf, 400.0, 90.0, 8.0]#, 1.0 
no_bins=30

# To plot multiple histograms across max SNRs
plt.figure()
for i in max_glitch_SNR_list:
    AET_dir= f"/fred/oz303/aboumerd/EMRI_Glitches/data_files/glitch_bg_AET/max_glitch_SNR_{i}/"
    background_AET_files= sorted(glob.glob(AET_dir + "*.h5"))
    opt_SNR_arr= np.zeros(len(background_AET_files))
    for k in range(len(background_AET_files)):
        #Load the AET variables
        background_idx= background_AET_files[k].split("/")[-1].split("_")[1]
        AET_filename= f"BG_{background_idx}_AET.h5"
        #Load data
        with h5py.File(AET_dir + AET_filename, 'r') as f:
            opt_SNR = f["SNR"][()]
        opt_SNR_arr[k]= opt_SNR
    bins_SNR=np.logspace(np.log10(np.min(opt_SNR_arr)), np.log10(np.max(opt_SNR_arr)), num=no_bins)
    plt.hist(opt_SNR_arr, bins=bins_SNR, alpha=0.7, label=f"Glitch SNRs $\leq$ {i}")

plt.title(f"Optimal SNRs of {len(background_AET_files)} glitch backgrounds")
plt.xlabel("Optimal SNR")
plt.ylabel("Counts")
plt.legend()
plt.xscale("log")
#Set the save directory to the max_SNR infinity one
plt.savefig(f"optimal_BG_SNRs_across_max_SNRs.pdf")
