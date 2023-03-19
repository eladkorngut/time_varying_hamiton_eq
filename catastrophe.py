# This program solves the Hamilton equations in case there is a sudden drop in the reproductive number

import numpy as np
import scipy.linalg as la
from scipy.integrate import odeint
from scipy.integrate import simps
import numdifftools as ndft
import matplotlib.pyplot as plt
import pickle
import os

def eq_points_exact(epsilon,beta,gamma):
    # for 2d bimodal case find the eq points
    epsilon_lam, epsilon_mu,lam = epsilon[0], epsilon[1],beta/gamma
    y1star=(-2*epsilon_mu*(1 + epsilon_lam*epsilon_mu)+ lam*(-1 + epsilon_mu)*(1 + (-1 + 2*epsilon_lam)*epsilon_mu)+ np.sqrt(lam**2 +epsilon_mu*(4*epsilon_mu +lam**2*epsilon_mu*(-2 +epsilon_mu**2) +4*epsilon_lam*(lam -(-2 + lam)*epsilon_mu**2) +4*epsilon_lam**2*epsilon_mu*(lam -(-1 + lam)*epsilon_mu**2))))/(4*lam*(-1 +epsilon_lam)*(-1 +epsilon_mu)*epsilon_mu)
    y2star=(lam + epsilon_mu*(-2 + 2*lam +lam*epsilon_mu+ 2*epsilon_lam*(lam +(-1 + lam)*epsilon_mu)) -np.sqrt(lam**2 +epsilon_mu*(4*epsilon_mu +lam**2*epsilon_mu*(-2 +epsilon_mu**2) +4*epsilon_lam*(lam -(-2 + lam)*epsilon_mu**2) +4*epsilon_lam**2*epsilon_mu*(lam -(-1 + lam)*epsilon_mu**2))))/(4*lam*(1 +epsilon_lam)*epsilon_mu*(1 + epsilon_mu))
    p1star=-np.log((lam + 2*epsilon_lam -epsilon_lam**2*(lam -2*epsilon_mu) +np.sqrt(lam**2 +4*lam*epsilon_lam*epsilon_mu- 4*(-2 + lam)*epsilon_lam**3*epsilon_mu+ epsilon_lam**4*(lam**2 -4*(-1 + lam)*epsilon_mu**2) +epsilon_lam**2*(4 - 2*lam**2 + 4*lam*epsilon_mu**2)))/(2*(1 +epsilon_lam)*(1 +epsilon_lam*epsilon_mu)))
    p2star= -np.log(-(lam - 2*epsilon_lam- epsilon_lam**2*(lam + 2*epsilon_mu) +np.sqrt(lam**2 + 4*lam*epsilon_lam*epsilon_mu- 4*(-2 + lam)*epsilon_lam**3*epsilon_mu+ epsilon_lam**4*(lam**2 - 4*(-1 + lam)*epsilon_mu**2) + epsilon_lam**2*(4 - 2*lam**2 + 4*lam*epsilon_mu**2)))/(2*(-1 + epsilon_lam)*(1 + epsilon_lam*epsilon_mu)))
    return y1star, y2star, 0.0, 0.0, p1star, p2star


def eq_hamilton_J(beta,time,epsilon=0.0,factor=1.0,gamma=1.0,duration=1.0,t0=5.0):
    def bimodal_mu_lam(q,t,lam,eps,factor,g,duration,t0):
        epsilon_lam, epsilon_mu = eps[0], eps[1]
        dy1_dt_sus_inf =lambda l: l * ((1 - epsilon_lam) * q[0] + (1 + epsilon_lam) * q[1]) * (1 - epsilon_mu) * (
                1 / 2 - q[0]) * np.exp(q[2]) - g * q[0] * np.exp(-q[2])
        dy2_dt_sus_inf = lambda l: l * ((1 - epsilon_lam) * q[0] + (1 + epsilon_lam) * q[1]) * (1 + epsilon_mu) * (
                1 / 2 - q[1]) * np.exp(q[3]) - g * q[1] * np.exp(-q[3])
        dtheta1_dt_sus_inf = lambda l: -l * (1 - epsilon_lam) * (
                (1 - epsilon_mu) * (1 / 2 - q[0]) * (np.exp(q[2]) - 1) + (1 + epsilon_mu) * (1 / 2 - q[1]) * (
                np.exp(q[3]) - 1)) + l * ((1 - epsilon_lam) * q[0] + (1 + epsilon_lam) * q[1]) * (
                                               1 - epsilon_mu) * (np.exp(q[2]) - 1) - g * (np.exp(-q[2]) - 1)
        dtheta2_dt_sus_inf = lambda l: -l * (1 + epsilon_lam) * (
                (1 - epsilon_mu) * (1 / 2 - q[0]) * (np.exp(q[2]) - 1) + (1 + epsilon_mu) * (1 / 2 - q[1]) * (
                np.exp(q[3]) - 1)) + l * ((1 - epsilon_lam) * q[0] + (1 + epsilon_lam) * q[1]) * (
                                    1 + epsilon_mu) * (np.exp(q[3]) - 1) - g * (np.exp(-q[3]) - 1)

        dy1_dt = dy1_dt_sus_inf(lam*factor) if t>t0 and t<t0+duration else dy1_dt_sus_inf(lam)
        dy2_dt= dy2_dt_sus_inf(lam*factor) if t>t0 and t<t0+duration else dy2_dt_sus_inf(lam)
        dp1_dt = dtheta1_dt_sus_inf(lam*factor) if t>t0 and t<t0+duration else dtheta1_dt_sus_inf(lam)
        dp2_dt = dtheta2_dt_sus_inf(lam*factor) if t>t0 and t<t0+duration else dtheta2_dt_sus_inf(lam)
        return np.array([dy1_dt, dy2_dt, dp1_dt, dp2_dt])
    y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy = eq_points_exact(epsilon, beta, gamma)
    return y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, bimodal_mu_lam, ndft.Jacobian(bimodal_mu_lam)

def postive_eigen_vec(J,q0,lam,eps,factor,gamma,duration,t0):
    # Find eigen vectors
    eigen_value, eigen_vec = la.eig(J(q0,0.0,lam,eps,factor,gamma,duration,t0))
    postive_eig_vec = []
    for e in range(np.size(eigen_value)):
        if eigen_value[e].real > 0:
            postive_eig_vec.append(eigen_vec[:, e].reshape(4, 1).real)
    return postive_eig_vec


def shoot(y1_0, y2_0, p1_0, p2_0, tshot, J,dq_dt,lam,epsilon,factor,gamma):
    q0 = (y1_0, y2_0, p1_0, p2_0)
    vect_J = lambda q, tshot: J(q0)
    # qsol = odeint(dq_dt, q0, tshot,atol=1.0e-20, rtol=1.0e-13, mxstep=1000000000, hmin=1e-30, Dfun=vect_J)
    qsol = odeint(dq_dt, q0, tshot,args=(lam,epsilon,factor,gamma,duration,t0),Dfun=vect_J)
    return qsol


def one_shot(shot_angle,lin_weight,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt,epsilon,lam,factor,gamma,duration,t0):
    q0 = (q_star[0] + radius * np.cos(shot_angle), q_star[1], 0+radius * np.sin(shot_angle), 0)
    postive_eig_vec = postive_eigen_vec(J, q0,lam,epsilon,factor,gamma,duration,t0)
    y1_i, y2_i, p1_i, p2_i = q0[0] + lin_weight * float(postive_eig_vec[0][0]) * one_shot_dt + (
                1 - lin_weight) * float(postive_eig_vec[1][0]) * one_shot_dt \
        , q0[1] + float(lin_weight * postive_eig_vec[0][1]) * one_shot_dt + (1 - lin_weight) * float(
        postive_eig_vec[1][1]) * one_shot_dt \
        , q0[2] + float(postive_eig_vec[0][2]) * one_shot_dt + (1 - lin_weight) * float(postive_eig_vec[1][2]) * one_shot_dt \
        , q0[3] + lin_weight * float(postive_eig_vec[0][3]) * one_shot_dt + (1 - lin_weight) * float(
        postive_eig_vec[1][3]) * one_shot_dt
    return shoot(y1_i, y2_i, p1_i, p2_i, final_time_path,J,shot_dq_dt,lam,epsilon,factor,gamma)


if __name__=='__main__':
    #Network Parameters
    beta, gamma = 1.6, 1.0
    epslam,epsmu=0.01,0.01
    epsilon =np.array([epsmu,epslam])
    lam = beta/(1+epslam*epsmu)
    factor =1.0
    abserr,relerr = 1.0e-20,1.0e-13
    # ODE parameters
    stoptime= 20.0
    numpoints = 100
    # Create the time samples for the output of the ODE solver
    t = np.linspace(0.0,stoptime,numpoints)
    dt = 1e-3
    # Radius around eq point, shooting angle and linear combination of the the eigenvectors
    r = 5e-5
    angle = 1.0
    duration,t0=1.0,5.0

    int_lin_combo = 0.979693127715
    y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, shot_dq_dt, J = eq_hamilton_J(lam,t,epsilon, factor, gamma)
    q_star = [y1_0, y2_0, p1_star_clancy, p2_star_clancy]
    path = one_shot(angle, int_lin_combo, q_star, r, t, dt, J, shot_dq_dt,epsilon,lam,factor,gamma,duration,t0)
    plt.plot((path[:,0]+path[:,1])/2,(path[:,2]+path[:,3])/2)
    plt.xlabel('w')
    plt.ylabel(r'$p_{w}$')
    plt.title(r'$p_{w}$ vs w for $R_{0}$='+str(beta))
    plt.show()
    print('This no love song')



