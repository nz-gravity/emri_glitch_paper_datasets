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

#Iterate plotting of argmax R at each SNR
R_argmax_counter = np.zeros_like(SD_ii)
plt.figure()
for SNR in max_glitch_SNR:
    #Load EMRI biases
    EMRI_biases_dir = f"/fred/oz303/aboumerd/EMRI_Glitches/data_files/EMRI_biases/max_glitch_SNR_{SNR}/"
    delta_theta_arr_file= EMRI_biases_dir + f"{fiducial_EMRI.label}_delta_theta_arr.npy"
    delta_theta_glitches= np.load(delta_theta_arr_file)
    #Calculate R vectors
    R_glitches = np.abs(delta_theta_glitches/SD_ii)
    #Plot a bar chart of the arg maxes
    R_argmax= np.argmax(R_glitches, axis=1)
    R_argmax_counter += np.bincount(R_argmax)

plt.pie(R_argmax_counter, labels=param_labels, radius=1.3, textprops={'fontsize': 14})
plt.savefig(f"{fiducial_EMRI.label}_argmax_R.pdf")
plt.close()