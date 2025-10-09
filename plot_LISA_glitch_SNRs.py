import numpy as np
import matplotlib.pyplot as plt
import h5py


LISA_SNR_file= "glitch_SNRs_dset.h5"

#Load LISA SNRs
with h5py.File(LISA_SNR_file, "r") as f:
    LISA_SNRs= f["SNR"][()]

#Load LPF SNRs
ordinary_data = np.loadtxt("/fred/oz303/aboumerd/software/glitch/data/2021-09-17-effective_glitch_parameters_ordinary.txt")
cold_data = np.loadtxt("/fred/oz303/aboumerd/software/glitch/data/2021-09-17-effective_glitch_parameters_cold.txt")
data= np.abs(np.concatenate([ordinary_data,cold_data]))
'''
The data is ordered as:
col. 1 := Beta
col. 2 := Alpha (+ve or -ve)
col. 3 := SNR
'''
LPF_SNRs = data[:,-1]


#Get some percentiles on the LISA SNRs
percentiles = {
    "10th" : np.percentile(LISA_SNRs, 10),
    "25th" : np.percentile(LISA_SNRs, 25),
    "50th" : np.percentile(LISA_SNRs, 50),
    "75th" : np.percentile(LISA_SNRs, 75),
    "90th" : np.percentile(LISA_SNRs, 90)
}

#Plot histogram of LISA and LPF glitch SNRs
fig, axs = plt.subplots(2,1, sharex=True)

#Make logarithmic bins since the data is better visualised that way
no_bins=40
#imposing limits on the bins since there are outlier SNRs
'''The LISA SNR has an extremely long tail, and a small number of extreme outliers.
So restrict the bins to the 1st and 99th percentile'''
bins_LISA=np.logspace(np.log10(np.percentile(LISA_SNRs,1)), np.log10(np.percentile(LISA_SNRs,99)), num=no_bins)#-2, 4
bins_LPF= np.logspace(np.log10(LPF_SNRs.min()), np.log10(LPF_SNRs.max()), num=no_bins)#, , np.log10(SNRs.min()), np.log10(SNRs.max()),

# fig.suptitle("SNRs of glitches observed in LPF")

axs[0].hist(LISA_SNRs, bins=bins_LISA, label="Using SciRDv1 PSD {$A_2,E_2$}")
# axs[0].hist(LPF_SNRs, bins=bins_LPF, label="Using LPF PSD", alpha=0.5)
axs[0].set_xscale("log")
axs[0].set_ylabel("Counts")
# axs[0].legend()

axs[1].hist(LISA_SNRs, bins=bins_LISA, cumulative=True, density=True, histtype="step")
# axs[1].hist(LPF_SNRs, bins=bins_LPF, cumulative=True, density=True, histtype="step", alpha=0.5)
# axs[1].plot(percentiles["10th"], 0.1 ,"s",label=f"10th percentile = {percentiles['10th']:.2f}" , color= "black")
# axs[1].plot(percentiles["25th"], 0.25, "p",label=f"25th percentile = {percentiles['25th']:.2f}" , color= "red")
# axs[1].plot(percentiles["50th"], 0.5, "P", label=f"50th percentile = {percentiles['50th']:.2f}", color= "green")
# axs[1].plot(percentiles["75th"], 0.75, "*", label=f"75th percentile = {percentiles['75th']:.2f}", color= "orange")
# axs[1].plot(percentiles["90th"], 0.9, "h", label=f"90th percentile = {percentiles['90th']:.2f}", color= "purple")
# axs[1].legend()


axs[1].set_ylabel("Cumulative density")
axs[1].set_xlabel("Network SNR")

# plt.legend()
plt.savefig("glitch_catalogue_SNR_hist.pdf")
plt.close()
