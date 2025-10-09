import numpy as np
import matplotlib.pyplot as plt

# Choose the levels of glitch mitigation we want to consider
max_glitch_SNR = [np.inf, 400.0, 90.0, 8.0]

EMRI_label= "Prograde_EMRI"#"Retrograde_EMRI"#"Strongfield_EMRI"

plt.figure()
for SNR in max_glitch_SNR:
    #Load R vectors
    R_glitches = np.load(f"{EMRI_label}_R_glitches_max_SNR_{SNR}.npy")
    #Calculate the max of each R vector
    R_max = np.max(R_glitches, axis=1)
    #Plot the CDF of R max
    no_bins=20
    bins= np.logspace(np.log10(R_max.min()), np.log10(R_max.max()), num=no_bins)
    plt.hist(R_max, label=f"Glitch SNRs $\\leq$ {SNR}", bins=bins, density=True, cumulative=True, histtype="step")

#Plot a vertical line for R=1
# plt.title(f"{fiducial_EMRI.label}: CDF of {R_glitches.shape[0]} instances of " + "$\max{[\\boldsymbol{\mathcal{R}}_i]}$")
plt.axvline(1, label="$\\mathcal{R}=1$", linestyle="--")
plt.xscale("log")
plt.legend()
plt.ylabel("Cumulative probability")
plt.xlabel("Parameter-wise $\\max{\\mathcal{(R)}}$")
plt.savefig(f"{EMRI_label}_max_R_CDF.pdf")
plt.close()
