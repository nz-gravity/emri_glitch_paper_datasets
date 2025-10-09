import numpy as np
import matplotlib.pyplot as plt
from h5py import File
from sklearn.utils import resample
from EMRI_settings import *

#Choose an EMRI
fiducial_EMRI = ProgradeEMRI()

# Choose the levels of glitch mitigation we want to consider
max_glitch_SNR = [np.inf, 400.0, 90.0, 8.0]#

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

#Iterate plotting of total biases over various glitch mitigation levels
plt.figure()
# plt.title(f"{fiducial_EMRI.label}\n Absolute glitch biases normalised by noise-induced uncertainty")
for SNR in max_glitch_SNR:
    #Load EMRI biases
    EMRI_biases_dir = f"/fred/oz303/aboumerd/EMRI_Glitches/data_files/EMRI_biases/max_glitch_SNR_{SNR}/"
    delta_theta_arr_file= EMRI_biases_dir + f"{fiducial_EMRI.label}_delta_theta_arr.npy"
    delta_theta_glitches= np.load(delta_theta_arr_file)
    #Calculate total bias: E(noise biases + glitch biases) = E(glitch biases)
    total_bias = np.mean(delta_theta_glitches, axis=0)
    #Take the magnitude of the bias, normalise by the noise-induced uncertainty
    normalised_total_bias= np.abs(total_bias/SD_ii)
    x_coords = np.arange(0,len(normalised_total_bias))
    #Scatter plot the normalised biases
    plt.scatter(x_coords, normalised_total_bias, marker=".", label=f"Glitch SNRs $\\leq$ {SNR}", zorder=2)

plt.xticks(x_coords, param_labels)
plt.grid(axis="x", zorder=1)
plt.yscale("log")
plt.legend()
plt.ylabel("$|\\beta_{\\text{glitches}}|$ / SD($\Delta \\theta_{\\text{noise}})$")
plt.xlabel("Parameter")
plt.savefig(f"{fiducial_EMRI.label}_glitch_biases.pdf")