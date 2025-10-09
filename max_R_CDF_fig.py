import numpy as np
import matplotlib.pyplot as plt
from h5py import File
from EMRI_settings import *

#Choose an EMRI
fiducial_EMRI = ProgradeEMRI()#StrongfieldEMRI()#RetrogradeEMRI()#ProgradeEMRI()

# Choose the levels of glitch mitigation we want to consider
max_glitch_SNR = [np.inf, 400.0, 90.0, 8.0]

#FM dir and filename
fisher_dir= "data_files/EMRI_fisher/"
fisher_fname= f"Fisher_{fiducial_EMRI.label}.h5"
fisher_file= fisher_dir+fisher_fname

#Load EMRI params
M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = (fiducial_EMRI.M,
                                                                                    fiducial_EMRI.mu,
                                                                                        fiducial_EMRI.a,
                                                                                        fiducial_EMRI.p0,
                                                                                            fiducial_EMRI.e0,
                                                                                            fiducial_EMRI.x0,
                                                                                                fiducial_EMRI.dist,
                                                                                                fiducial_EMRI.qS,
                                                                                                    fiducial_EMRI.phiS,
                                                                                                    fiducial_EMRI.qK,
                                                                                                        fiducial_EMRI.phiK,
                                                                                                        fiducial_EMRI.Phi_phi0,
                                                                                                            fiducial_EMRI.Phi_theta0,
                                                                                                            fiducial_EMRI.Phi_r0)

#Omitting the params that we don't estimate!
params = [M, mu, a, p0, e0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_r0]#x0, Phi_theta0,

#Define all param names, labels and units
param_names = ['M','mu','a','p0','e0','dist', 'qS','phiS','qK','phiK','Phi_phi0','Phi_r0']#'Y0', 'Phi_theta0',
param_labels = ['$M$','$\\mu$','$a$','$p_0$','$e_0$','$d_L$', '$\\theta_S$','$\\phi_S$','$\\theta_K$','$\\phi_K$','$\\Phi_{\\phi_0}$','$\\Phi_{r_0}$']#'$Y_0$', '$\\Phi_{\\theta_0}$',
params_units = ['$M_\\odot$', '$M_\\odot$','','','','Gpc', 'rad','rad','rad','rad','rad','rad']#'', 'rad'

#Load FM and calculate noise-induced covariance
with File(fisher_file,"r") as f:
    fisher= np.array(f['Fisher'][()])

noise_covariance= np.linalg.inv(fisher)
noise_covariance_ii= np.diag(noise_covariance)
SD_ii = noise_covariance_ii**0.5


#Iterate plotting of CDF of max R over various glitch mitigation levels
plt.figure()
for SNR in max_glitch_SNR:
    #Load EMRI errors
    EMRI_errors_dir = f"data_files/EMRI_errors/max_glitch_SNR_{SNR}/"
    delta_theta_arr_file= EMRI_errors_dir + f"{fiducial_EMRI.label}_delta_theta_arr.npy"
    delta_theta_glitches= np.load(delta_theta_arr_file)
    #Calculate R vectors
    R_glitches = np.abs(delta_theta_glitches/SD_ii)
    #Calculate the parameter-wise max of each R vector
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
plt.savefig(f"{fiducial_EMRI.label}_max_R_CDF.pdf")
plt.close()