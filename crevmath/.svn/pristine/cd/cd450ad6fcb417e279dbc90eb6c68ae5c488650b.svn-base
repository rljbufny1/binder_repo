import numpy as np

def AofT(T):
    """
    Description:
        Calculates the flow coefficient, A, from the Cuffey and Patterson (2010) piecewise function.
        Equation 3.35 on page 72.
        Author: Ben Reynolds (July 4, 2024)
    Arguments:
        T: temperature field as numpy array (C)
    Returns:
        The flow factor, A (s^-1 Pa^-3)
    """
    A_star = 3.5e-25; # s^-1 Pa^-3
    T_star = 263; # K
    Q_plus = 115e3; # J*mol^-1
    Q_min = 6e4; # J*mol^-1
    R = 8.314; # J*mol^-1*K^-1

    T_K = T + 273.15; # temp in Kelvin

    # calculate entire field with both parts of piecewise functions
    A_eq_m = A_star*np.exp(-Q_min/R*(1/T_K - 1/T_star));
    A_eq_p = A_star*np.exp(-Q_plus/R*(1/T_K - 1/T_star));

    # set flow factor with fields from each piecewise equation calc according to threshold temperature
    A = A_eq_m;
    A[1/T_K < 1/T_star] = A_eq_p[1/T_K < 1/T_star];
    
    return A


def eig_2D_sym(a_xx, a_yy, a_xy):
    # calculates eigen values for a 2d symmetric matrix
    
    a_1 = (a_xx + a_yy)/2 + np.sqrt(((a_xx-a_yy)/2)**2 + a_xy**2)
    a_2 = (a_xx + a_yy)/2 - np.sqrt(((a_xx-a_yy)/2)**2 + a_xy**2)
    
    return a_1, a_2


def eff_strain_rate(e_xx, e_yy, e_zz, e_xy, e_xz, e_yz):
    # calculaes effective strain rate from the six unique components 

    e_eff = (0.5*(e_yy**2 + e_xx**2 + e_zz**2) + e_xy**2 + e_xz**2 + e_yz**2)**0.5

    return e_eff


def zero_stress_basal(rho_i, rho_pw, g, Rxx_b, H_ab):
    # calculate a basal crevasse height assuming a constant resistive stress
    
    db = rho_i/(rho_pw-rho_i) * (Rxx_b/(rho_i*g)-H_ab)

    return db


def nyeCrevassesAF(X, Y, VX, VY, surf, thk, Tsurf, Tbase, rho_i, rho_pw, g, RxxCalc):    
    """
    Description:
        This function takes in velocity, surface elevation, ice thickness, temperature
        and user specified stress calculation version. It calculates strain rates from 
        velocity using central differences. Then it calculates deviatoric stresses from 
        strain rates with Cuffey and Patterson (2010) rheology and effective strain rate 
        based on user specified calculation. Finally it calculates resistive stress applied
        to crevasses assuming flow direction or maximum principal direction and with or without
        crevasse parallel stress and uses that resistive with the Nye crevasse formulation to
        get surface and basal crevasse depths/heights. 
        Author: Ben Reynolds (July 4, 2024)
    Arguments:
        NOTE: all fields should be input as 2D numpy arrays. Code has been tested with numpy
        arrays that are aligned with increasing x and y coordinates. A way of checking this is
        plotting the array with plt.pcolor or plt.contourf and seeing that the orientations is
        correct. (The gradient function is the concern and it should still work by using negative
        grid spacing, but have NOT checked).
        x: X points as 1D numpy array [m]
        y: Y points as 1D numpy array [m]
        Vx: X velocity component [m/s]
        Vy: Y velocity component [m/s]
        surf: surface elevation [m]
        thk: ice thickness [m]
        Tsurf: surface temperature [C]
        Tbase: base temperature [C]
        RxxCalc: resistive stress calculation choice input as a capital letter.
            Options: A,B,C,D,E,F. 
    Returns:
        ds: surface crevasse depth [m]
        db: basal crevasse height [m]
        Rxx_s: surface resistive stress from the chosen calculation [Pa]
        tau_xx_s: surface deviatoric stress perpendicular to the crevasse from the chosen calculation [Pa]
        tau_yy_s: surface deviatoric stress parallel to the crevasse from the chosen calculation [Pa]
        Rxx_b: base resistive stress [Pa]
        tau_xx_b: base deviatoric stress perpendicular to the crevasse [Pa]
        tau_yy_b: base deviatoric stress parallel to the crevasse [Pa]
    Stress calculations:
        A: flow direction, no effective strain rate, no crevasse parallel deviatoric stress
        B: max prin direction, no effective strain rate, no crevasse parallel deviatoric stress
        C: max prin direction, planar effective strain rate, no crevasse parallel deviatoric stress
        D: max prin direction, full effective strain rate, no crevasse parallel deviatoric stress
        E: max prin direction, planar effective strain rate, includes crevasse parallel deviatoric stress
        F: max prin direction, full effective strain rate, includes crevasse parallel deviatoric stress
    """

    # using cuffey and patterson (2010) rheology lookup with n = 3
    n = 3

    # select stress direction, effective strain rate, and crev parallel stress based on calculation version
    if RxxCalc == 'A':
        stressDir = 'flow'
        effStrainRate = 'none'
        includeCrevPar = 'no'
    elif RxxCalc == 'B':
        stressDir = 'maxPrin'
        effStrainRate = 'none'
        includeCrevPar = 'no'
    elif RxxCalc == 'C':
        stressDir = 'maxPrin'
        effStrainRate = 'planar'
        includeCrevPar = 'no'
    elif RxxCalc == 'D':
        stressDir = 'maxPrin'
        effStrainRate = 'full'
        includeCrevPar = 'no'
    elif RxxCalc == 'E':
        stressDir = 'maxPrin'
        effStrainRate = 'planar'
        includeCrevPar = 'yes'
    elif RxxCalc == 'F':
        stressDir = 'maxPrin'
        effStrainRate = 'full'
        includeCrevPar = 'yes'
    else:
        print('RxxCalc argument must be letter A through F capitalized')

    # get grid spacing for gradient function
    del_X = X[1] - X[0]
    del_Y = Y[1] - Y[0]

    # numpy gradient function returns gradientient in rows and then columns direction for 2D input
    # l_ij denotes components of the velocity gradient tensor and e_ij denotes components of the strain
    # rate tensor. l_xx and l_yy are same as e_xx and e_yy (because on diagonal), but e_xy is 1/2*(l_xy + l_yx).
    # See strain-rate tensor wiki.
    l_XY, e_XX = np.gradient(VX, del_X, del_Y)
    e_YY, l_YX = np.gradient(VY, del_X, del_Y)

    e_XY = 0.5 * (l_XY + l_YX) # the shear strain rate 

    # calculate strain rates in crevasse direction according to specified direction
    if stressDir == 'flow':
        theta_f = np.arctan2(VY, VX)
        e_xx = (e_XX+e_YY)/2 + ((e_XX-e_YY)/2)*np.cos(2*theta_f) + e_XY*np.sin(2*theta_f)
        e_yy = (e_XX+e_YY)/2 - ((e_XX-e_YY)/2)*np.cos(2*theta_f) - e_XY*np.sin(2*theta_f)
        e_xy = -((e_XX-e_YY)/2)*np.sin(2*theta_f) + e_XY*np.cos(2*theta_f)
    elif stressDir == 'maxPrin':
        e_xx = (e_XX + e_YY)/2 + np.sqrt(((e_XX-e_YY)/2)**2 + e_XY**2)
        e_yy = (e_XX + e_YY)/2 - np.sqrt(((e_XX-e_YY)/2)**2 + e_XY**2)
        e_xy = 0 # if you rotate to max prin dir shear is by def 0
    
    # calculate effective strain rate with specified method
    if effStrainRate == 'none':
        e_eff = np.abs(e_xx) # works because e_zz would be assuemd to be -e_xx if neglecting strain rate 
    elif effStrainRate == 'planar':
        e_zz = 0.0 # neglect vertical strain rate
        e_eff = (0.5*(e_yy**2 + e_xx**2 + e_zz**2) + e_xy**2)**0.5;
    elif effStrainRate == 'full':
        e_zz = -e_xx - e_yy # get vertical strain rate with continuity
        e_eff = (0.5*(e_yy**2 + e_xx**2 + e_zz**2) + e_xy**2)**0.5;

    # rigidity is B = A^(-1/n)
    B_s = AofT(Tsurf)**(-1/n)
    B_b = AofT(Tbase)**(-1/n)
    
    # calculate deviatoric surfacae and base stress in crev perp (xx) and parallel (yy)
    tau_xx_s = B_s * e_eff**((1-n)/n) * e_xx
    tau_yy_s = B_s * e_eff**((1-n)/n) * e_yy
    tau_xx_b = B_b * e_eff**((1-n)/n) * e_xx
    tau_yy_b = B_b * e_eff**((1-n)/n) * e_yy

    # include crevasse parallel stress if specified by calculation
    if includeCrevPar == 'yes':
        Rxx_s = 2*tau_xx_s + tau_yy_s
        Rxx_b = 2*tau_xx_b + tau_yy_b
    elif includeCrevPar == 'no':
        Rxx_s = 2*tau_xx_s
        Rxx_b = 2*tau_xx_b

    # height above buoyancy for basal crevasse calc
    H_ab = thk - (rho_pw/rho_i) * (thk-surf)
    
    # calculate surface and basal crevasse heights
    ds = Rxx_s/(rho_i * g)
    db = rho_i/(rho_pw-rho_i) * (Rxx_b/(rho_i*g)-H_ab)
    
    # clip negative crevasse sizes to 0
    ds[ds<0] = 0
    db[db<0] = 0

    return ds, db, Rxx_s, tau_xx_s, tau_yy_s, Rxx_b, tau_xx_b, tau_yy_b


def calcPenRatio(ds, db, thk):
    penRatio = (ds+db)/thk
    penRatio[penRatio>1] = 1.0
    return penRatio