import numpy as np
import matplotlib.pyplot as plt
from h5py import File
from sklearn.utils import resample
from EMRI_settings import *

#Set a random seed
seed=1234
np.random.seed(seed)

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

#Generate samples of noise-induced biases using the FM-derived covariance
delta_theta_noise = np.random.multivariate_normal(np.zeros(12), noise_covariance, 10000)

#Iterate plotting of total precisions over various glitch mitigation levels
plot_dir= f"data_files/EMRI_errors/max_glitch_SNR_inf/"
plt.figure()
# plt.title(f"{fiducial_EMRI.label}\n Uncertainty ratios for many glitch+noise backgrounds")
for SNR in max_glitch_SNR:
    #Load EMRI biases
    EMRI_errors_dir = f"data_files/EMRI_errors/max_glitch_SNR_{SNR}/"
    delta_theta_arr_file= EMRI_errors_dir + f"{fiducial_EMRI.label}_delta_theta_arr.npy"
    delta_theta_glitches= np.load(delta_theta_arr_file)
    '''Let's do something funky: resample the glitch-induced biases and add them to the noise-induced biases'''
    #Resample the glitch biases
    resampled_delta_theta_glitches = resample(delta_theta_glitches, n_samples=delta_theta_noise.shape[0], random_state=seed)
    #Calculate total error across all parameters
    delta_theta_total= resampled_delta_theta_glitches + delta_theta_noise
    #Calculate_total_precision
    total_precision = np.std(delta_theta_total, axis=0)
    normalised_total_precision= total_precision/SD_ii
    x_coords = np.arange(0,len(normalised_total_precision))
    #Scatter plot the params
    plt.scatter(x_coords, normalised_total_precision, marker=".", label=f"Glitch SNRs $\\leq$ {SNR}", zorder=2)

plt.xticks(x_coords, param_labels)
plt.grid(axis="x", zorder=1)
plt.legend()
plt.ylabel("$SD(\Delta \\theta_{\\text{total}})$ / $SD(\Delta \\theta_{\\text{noise}})$")
plt.xlabel("Parameter")
plt.savefig(f"{fiducial_EMRI.label}_total_precisions.pdf")