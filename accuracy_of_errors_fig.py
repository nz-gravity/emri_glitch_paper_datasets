import numpy as np
from eryn.backends import HDFBackend as eryn_HDF_Backend
import matplotlib.pyplot as plt
from sklearn.utils import resample
import corner
import os
import warnings
import matplotlib.lines as mlines
import pickle
from h5py import File
from matplotlib.font_manager import FontProperties

#Set a random seed
seed=1234
np.random.seed(seed)

#specify which run we want corresponding to some level of glitch mitigation
max_glitch_SNR= [np.inf, 400.0, 90.0, 8.0]

#Choose a glitch bg
glitch_bg_idx=0

#Glitchless samples dir and filenames
glitchless_samples_dir = f"data_files/EMRI_mcmc_samples/max_glitch_SNR_inf/"
samples_filename = "Prograde_EMRI_M-1e06-mu-10-a-0_998-p0-7_73-e0-0_73-SNR-80.h5"#"Retrograde_EMRI_M-1e05-mu-10-a--0_500-p0-26_19-e0-0_80-SNR-80.h5"#"Prograde_EMRI_M-1e06-mu-10-a-0_998-p0-7_73-e0-0_73-SNR-80.h5"#f"Strongfield_EMRI_M-1e07-mu-10-a-0_998-p0-2_12-e0-0_42-SNR-80.h5"
glitchy_samples_filename = f"BG_{glitch_bg_idx:0>4}_PLUS_{samples_filename}"
glitchless_params_filename = f"PARAMS_{samples_filename}"

#Get EMRI label
EMRI_label = samples_filename.split("_M-")[0]

#FM dir and filename
fisher_dir= "data_files/EMRI_fisher/"
fisher_fname= f"Fisher_{EMRI_label}.h5"
fisher_file= fisher_dir+fisher_fname

#load dict of EMRI params
params_file= glitchless_samples_dir + glitchless_params_filename

with open(params_file, 'rb') as f:
    params_dict = pickle.load(f)

true_vals = np.array([params_dict["M"], params_dict["mu"], params_dict["a"], params_dict["p0"], params_dict["e0"], 
                       params_dict["dist"], params_dict["qS"], params_dict["phiS"], params_dict["qK"], params_dict["phiK"],
                       params_dict["Phi_phi0"], params_dict["Phi_r0"]])#params_dict["x0"],params_dict["Phi_theta0"],

param_labels = ['$M$','$\\mu$','$a$','$p_0$','$e_0$','$d_L$', '$\\theta_S$','$\\phi_S$','$\\theta_K$','$\\phi_K$','$\\Phi_{\\phi_0}$','$\\Phi_{r_0}$']#'$Y_0$', '$\\Phi_{\\theta_0}$',

#Load FM and calculate noise-induced covariance
with File(fisher_file,"r") as f:
    fisher= np.array(f['Fisher'][()])

noise_covariance= np.linalg.inv(fisher)
noise_covariance_ii= np.diag(noise_covariance)
SD_ii = noise_covariance_ii**0.5

#Iterate over varying levels of glitch mitigation
glitchy_burnin=2000#2000#1000#2000

plt.figure()
for i in max_glitch_SNR:
    #Glitchy samples dirs and filenames
    glitchy_samples_dir = f"data_files/EMRI_mcmc_samples/max_glitch_SNR_{i}/"
    glitchy_params_filename = f"PARAMS_{glitchy_samples_filename}"
    #Load glitchy samples
    glitchy_file= glitchy_samples_dir + glitchy_samples_filename
    reader_2 = eryn_HDF_Backend(glitchy_file,read_only = True)
    N_iterations = reader_2.get_chain()['model_0'].shape[0]
    N_temps = reader_2.get_chain()['model_0'].shape[1]
    N_walkers = reader_2.get_chain()['model_0'].shape[2]
    N_params = reader_2.get_chain()['model_0'].shape[-1]
    glitchy_samples_after_burnin = [reader_2.get_chain(discard = glitchy_burnin)['model_0'][:,i].reshape(-1,N_params) 
                        for i in range(N_temps)]  # Take true chain]
    glitchy_samples_corner = np.column_stack(glitchy_samples_after_burnin)
    #Calculate the MCMC-derived errors
    delta_theta_MCMC= glitchy_samples_corner.mean(axis=0)-true_vals#glitchless_samples_corner.mean(axis=0)
    #Calculate the FM-derived errors
    EMRI_biases_dir = f"data_files/EMRI_errors/max_glitch_SNR_{i}/"
    delta_theta_arr_file= EMRI_biases_dir + f"{EMRI_label}_delta_theta_arr.npy"
    delta_theta_FM= np.load(delta_theta_arr_file, allow_pickle=True)
    delta_theta_FM= delta_theta_FM[glitch_bg_idx,:]
    #Calculate the absolute difference in errors relative to the noise uncertainty
    relative_error = np.abs((delta_theta_FM-delta_theta_MCMC)/SD_ii)
    x_coords = np.arange(0,len(relative_error))
    #Scatter plot the params
    plt.scatter(x_coords, relative_error, marker=".", label=f"Glitch SNRs $\\leq$ {i}", zorder=2)

#Finishing touches
plt.xticks(x_coords, param_labels)
plt.grid(axis="x", zorder=1)
plt.yscale("log")
plt.xlabel("Parameter")
plt.ylabel("$(\hat\\theta_{\\text{FM}}-\hat\\theta_{\\text{MCMC}})/SD(\\Delta\\theta_{\\text{noise}})$")#$RE(\hat\\theta_{FM},\hat\\theta_{MCMC})$
# plt.title(f"{EMRI_label}: relative errors obtained due to glitch BG {glitch_bg_idx}")
plt.legend()
plt.savefig(f"{EMRI_label}_relative_biases_comparison.pdf")

