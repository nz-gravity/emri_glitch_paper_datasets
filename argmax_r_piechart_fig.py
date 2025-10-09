import numpy as np
import matplotlib.pyplot as plt

# Choose the levels of glitch mitigation we want to consider
max_glitch_SNR = [np.inf, 400.0, 90.0, 8.0]

EMRI_label= "Prograde_EMRI"#"Retrograde_EMRI"#"Strongfield_EMRI"
param_labels = ['$M$','$\\mu$','$a$','$p_0$','$e_0$','$d_L$', '$\\theta_S$','$\\phi_S$','$\\theta_K$','$\\phi_K$','$\\Phi_{\\phi_0}$','$\\Phi_{r_0}$']#'$Y_0$', '$\\Phi_{\\theta_0}$',

R_argmax_counter = np.zeros_like(12)#Hardcoded
plt.figure()
for SNR in max_glitch_SNR:
    #Load R vectors
    R_glitches = np.load(f"{EMRI_label}_R_glitches_max_SNR_{SNR}.npy")
    #Make a pie chart of the arg maxes
    R_argmax= np.argmax(R_glitches, axis=1)
    R_argmax_counter += np.bincount(R_argmax)

plt.pie(R_argmax_counter, labels=param_labels, radius=1.3, textprops={'fontsize': 14})
plt.savefig(f"{EMRI_label}_argmax_R.pdf")
plt.close()
