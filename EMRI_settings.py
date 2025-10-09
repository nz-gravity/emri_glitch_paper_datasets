import numpy as np

'''Rework: We're going to store the EMRI parameters in classes relating to each EMRI instead'''

class ProgradeEMRI:
    def __init__(self):
        #Store a label to describe the EMRI
        self.label = "Prograde_EMRI"
        #Store the waveform type we're using (this does affect the EMRI's SNR)
        self.waveform_model = "FastKerrEccentricEquatorialFlux"
        #Store all EMRI parameters as attributes
        self.M = 1e6
        self.mu = 1e1
        self.a = 0.998
        self.p0 = 7.728
        self.e0 = 0.730
        self.x0 = 1.0
        self.dist = 1.3242#2.204
        self.qS = 0.8
        self.phiS = 2.2
        self.qK = 1.6
        self.phiK = 1.2
        self.Phi_phi0 = 2.0
        self.Phi_theta0 = 0.0
        self.Phi_r0 = 3.0
        #Also the specifcy the sampling interval required for this EMRI
        self.dt = 5.0
    
class StrongfieldEMRI:
    def __init__(self):
        #Store a label to describe the EMRI
        self.label = "Strongfield_EMRI"
        #Store the waveform type we're using (this does affect the EMRI's SNR)
        self.waveform_model = "FastKerrEccentricEquatorialFlux"
        #Store all EMRI parameters as attributes
        self.M = 1e7
        self.mu = 1e1
        self.a = 0.998
        self.p0 = 2.120
        self.e0 = 0.425
        self.x0 = 1.0
        self.dist = 1.3536#3.590
        self.qS = 0.8
        self.phiS = 2.2
        self.qK = 1.6
        self.phiK = 1.2
        self.Phi_phi0 = 2.0
        self.Phi_theta0 = 0.0
        self.Phi_r0 = 3.0
        #Also the specifcy the sampling interval required for this EMRI
        self.dt = 5.0

class RetrogradeEMRI:
    def __init__(self):
        #Store a label to describe the EMRI
        self.label = "Retrograde_EMRI"
        #Store the waveform type we're using (this does affect the EMRI's SNR)
        self.waveform_model = "FastKerrEccentricEquatorialFlux"
        #Store all EMRI parameters as attributes
        self.M = 1e5
        self.mu = 1e1
        self.a = -0.500
        self.p0 = 26.192
        self.e0 = 0.800
        self.x0 = 1.0
        self.dist = 0.38055#1.0805
        self.qS = 0.8
        self.phiS = 2.2
        self.qK = 1.6
        self.phiK = 1.2
        self.Phi_phi0 = 2.0
        self.Phi_theta0 = 0.0
        self.Phi_r0 = 3.0
        #Also the specifcy the sampling interval required for this EMRI
        self.dt = 2.0


#Example: M ~ O(1e7), mu ~ O(1e2), SNR 82
# '''Note that for Fast eccentric Kerr model, x0 can only be 1.'''
# M = 1e7; mu = 800.0; a = 0.6; p0 = 8.8173; e0 = 0.05; 
# x0 = 1; dist = 2.69; qS = 1.0; phiS = 0.9; qK = 1.0; phiK = 0.8; 
# Phi_phi0 = 1.0; Phi_theta0 = 1.5; Phi_r0 = 3.0; 

#Example: M ~ O(1e7), mu ~ O(1e2), SNR 82
# M = 1e7; mu = 800.0; a = 0.6; p0 = 9.31; e0 = 0.05; 
# xI = np.pi/3; theta= 1.0; phi= 0.9


# Y0 = np.cos(iota0); dist = 1.25; #0.11
# qS = 1.0; phiS = 0.9; qK = 1.0; phiK = 0.8; 
# Phi_phi0 = 1.0; Phi_theta0 = 1.5; Phi_r0 = 3.0;

"""#Example: M ~ O(1e7), mu ~ O(1e2), SNR 82

#Example: M ~ O(1e5), mu ~ O(1e1), SNR 81
# M = 5e5; mu = 5.0; a = 0.4; p0 = 11.5; e0 = 0.15; 
# iota0 = -np.pi/3; Y0 = np.cos(iota0); dist = 0.73; 
# qS = 0.8; phiS = 1.2; qK = 0.5; phiK = 1.4; 
# Phi_phi0 = 0.2; Phi_theta0 = 1.5; Phi_r0 = 0.7; 

# Kerr Parameters -- textbook params, FM works
# M = 1e6; mu = 10.0; a = 0.9; p0 = 9.2; e0 = 0.2;
# iota0 = 0.8; Y0 = np.cos(iota0); dist = 2.0; 
# qS = 1.5; phiS = 0.7; qK = 1.2; phiK = 0.6; 
# Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0; 

# Kerr Parameters -- Plunging case, FM almost works -- out by a little. Plunging.
# M = 1e6; mu = 10.0; a = 0.9; p0 = 9.1; e0 = 0.2;
# iota0 = 0.8; Y0 = np.cos(iota0); dist = 2.0; 
# qS = 1.5; phiS = 0.7; qK = 1.2; phiK = 0.6; 
# Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0; 

# Kerr Parameters -- Not plunging, FM is a dodgy approximation here. Looks "OK"

# M = 2e6; mu = 20.0; a = 0.5; p0 = 8.44; e0 = 0.3;
# iota0 = 0.8; Y0 = np.cos(iota0); dist = 4.0; 
# qS = 1.5; phiS = 0.7; qK = 1.2; phiK = 0.6; 
# Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0; 

# Kerr Parameters -- Ramp up SNR of the point above...
# Seems normal distribution. SNR upgrade. This works with FM! 
# M = 2e6; mu = 20.0; a = 0.5; p0 = 8.44; e0 = 0.3;
# iota0 = 0.8; Y0 = np.cos(iota0); dist = 1.0; 
# qS = 1.5; phiS = 0.7; qK = 1.2; phiK = 0.6; 
# Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0; 

# Kerr Parameters -- Doesn't seem to work? 
# M = 1e6; mu = 20.0; a = 0.3; p0 = 11.09; e0 = 0.6;
# iota0 = 0.2; Y0 = np.cos(iota0); dist = 1.0; 
# qS = 1.5; phiS = 0.7; qK = 1.2; phiK = 0.6; 
# Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0; 

# Kerr parameters -- SNR 123, strange masses...
# M = 1e7; mu = 500.0; a = 0.99; p0 = 7.95; e0 = 0.2;
# iota0 = np.pi/3; Y0 = np.cos(iota0); dist = 1.0; 
# qS = 1.5; phiS = 0.2; qK = 1.2; phiK = 0.6; 
# Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0; 

# Kerr Parameters -- Not plunging, new point Seems to work?
# M = 1e6; mu = 20.0; a = 0.3; p0 = 11.09; e0 = 0.6;
# iota0 = 0.2; Y0 = np.cos(iota0); dist = 4.0; 
# qS = 1.5; phiS = 0.7; qK = 1.2; phiK = 0.6; 
# Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0; 

# Kerr Parameters -- Pretty crazy point -- This works!
# M = 5e5; mu = 5.0; a = 0.95; p0 = 10.851; e0 = 0.3;
# iota0 = np.pi/3; Y0 = np.cos(iota0); dist = 3.0; 
# qS = 0.8; phiS = 0.6; qK = 0.8; phiK = 0.4; 
# Phi_phi0 = 2.5; Phi_theta0 = 2.5; Phi_r0 = 2.5;
#  
# New Extreme parameters -- test with Fisher matrix -- need to test this.
# M = 5e6; mu = 10; a = 0.8; p0 = 5.40; e0 = 0.5; Y0 = np.cos(np.pi/3)
# dist = 1.0; qS = 1.5; phiS = 0.7; qK = 1.2; phiK = 0.6
# Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0
"""

# Waveform params
delta_t = 10.0;  # Sampling interval [seconds]
T = 2.0     # Evolution time [years]

mich = False #mich = True implies output in hI, hII long wavelength approximation


# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)





