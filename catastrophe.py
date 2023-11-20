# This program solves the Hamilton equations in case there is a sudden drop in the reproductive number

import numpy as np
import scipy.linalg as la
from scipy.integrate import odeint
from scipy.integrate import simps
import numdifftools as ndft
import matplotlib.pyplot as plt
import pickle
import os

peak_location = lambda lam: -2*((( (lam-1)*(1-12*lam+3*lam**2)+8*(lam**2)*np.log(lam) )/(4*lam**3))-(1-1/lam)**2)/(1-1/lam)**2/2


def eq_points_exact(epsilon,beta,gamma):
    # for 2d bimodal case find the eq points
    epsilon_lam, epsilon_mu,lam = epsilon[0], epsilon[1],beta/gamma
    y1star=(-2*epsilon_mu*(1 + epsilon_lam*epsilon_mu)+ lam*(-1 + epsilon_mu)*(1 + (-1 + 2*epsilon_lam)*epsilon_mu)+ np.sqrt(lam**2 +epsilon_mu*(4*epsilon_mu +lam**2*epsilon_mu*(-2 +epsilon_mu**2) +4*epsilon_lam*(lam -(-2 + lam)*epsilon_mu**2) +4*epsilon_lam**2*epsilon_mu*(lam -(-1 + lam)*epsilon_mu**2))))/(4*lam*(-1 +epsilon_lam)*(-1 +epsilon_mu)*epsilon_mu)
    y2star=(lam + epsilon_mu*(-2 + 2*lam +lam*epsilon_mu+ 2*epsilon_lam*(lam +(-1 + lam)*epsilon_mu)) -np.sqrt(lam**2 +epsilon_mu*(4*epsilon_mu +lam**2*epsilon_mu*(-2 +epsilon_mu**2) +4*epsilon_lam*(lam -(-2 + lam)*epsilon_mu**2) +4*epsilon_lam**2*epsilon_mu*(lam -(-1 + lam)*epsilon_mu**2))))/(4*lam*(1 +epsilon_lam)*epsilon_mu*(1 + epsilon_mu))
    p1star=-np.log((lam + 2*epsilon_lam -epsilon_lam**2*(lam -2*epsilon_mu) +np.sqrt(lam**2 +4*lam*epsilon_lam*epsilon_mu- 4*(-2 + lam)*epsilon_lam**3*epsilon_mu+ epsilon_lam**4*(lam**2 -4*(-1 + lam)*epsilon_mu**2) +epsilon_lam**2*(4 - 2*lam**2 + 4*lam*epsilon_mu**2)))/(2*(1 +epsilon_lam)*(1 +epsilon_lam*epsilon_mu)))
    p2star= -np.log(-(lam - 2*epsilon_lam- epsilon_lam**2*(lam + 2*epsilon_mu) +np.sqrt(lam**2 + 4*lam*epsilon_lam*epsilon_mu- 4*(-2 + lam)*epsilon_lam**3*epsilon_mu+ epsilon_lam**4*(lam**2 - 4*(-1 + lam)*epsilon_mu**2) + epsilon_lam**2*(4 - 2*lam**2 + 4*lam*epsilon_mu**2)))/(2*(-1 + epsilon_lam)*(1 + epsilon_lam*epsilon_mu)))
    return y1star, y2star, 0.0, 0.0, p1star, p2star


hamiltonian = lambda y1,y2,p1,p2,epsilon,R0:((-1 + np.exp(-p1)) * y1 + (-1 + np.exp(-p2)) * y2 -(R0 * (-2 + y1 + y2 +
            np.exp(p1) * (-1 + y1) * (-1 + epsilon) - y1 * epsilon + y2 * epsilon -np.exp(p2) * (-1 + y2) *
            (1 + epsilon)) * (y1 * (-1 + epsilon) - y2 * (1 + epsilon))) /(2 * (1 + epsilon ** 2))) / 2

def eq_points_mj(epsilon,R0):
    y1star = (-2*(epsilon + epsilon**3) + R0*(-1 + epsilon)*(1 + epsilon*(-1 + 2*epsilon)) +
              np.sqrt(R0**2*(-1 + epsilon**2)**2 + 4*(epsilon + epsilon**3)**2 -4*R0*epsilon**2*(-1 + epsilon**4)))\
             /(2*R0*(-1 + epsilon)**2*epsilon)
    y2star = -(-(R0*(1 + epsilon)*(1 + epsilon + 2*epsilon**2)) + 2*(epsilon + epsilon**3) +
               np.sqrt(R0**2*(-1 + epsilon**2)**2 + 4*(epsilon + epsilon**3)**2 -4*R0*epsilon**2*(-1 + epsilon**4)))\
             /(2*R0*epsilon*(1 + epsilon)**2)
    p1star = np.log(2) - np.log((R0 + 2*epsilon - R0*epsilon**2 + 2*epsilon**3 +np.sqrt(R0**2*(-1 + epsilon**2)**2 +
            4*(epsilon + epsilon**3)**2 -4*R0*epsilon**2*(-1 + epsilon**4)))/(1 + epsilon + epsilon**2 + epsilon**3))
    p2star = np.log(2) - np.log((R0*(-1 + epsilon**2) + 2*(epsilon + epsilon**3) -np.sqrt(R0**2*(-1 + epsilon**2)**2 +
              4*(epsilon + epsilon**3)**2 -4*R0*epsilon**2*(-1 + epsilon**4)))/((-1 + epsilon)*(1 + epsilon**2)))
    return y1star,y2star,0.0,0.0,p1star,p2star

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
    y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy = eq_points_exact(epsilon, beta*(1+epsilon[0]*epsilon[1]), gamma)
    return y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, bimodal_mu_lam, ndft.Jacobian(bimodal_mu_lam)


def eq_motion_mj(lam,t0,tf,xi,epsilon):
    # Hamilton eq for 2d case
    dy1dt_mj = lambda q, R0: -q[0]/(2*np.exp(q[2])) - (np.exp(q[2])*R0*(-1 + q[0])*(-1 + epsilon)*(q[0]*(-1 + epsilon) -
                        q[1]*(1 + epsilon)))/(4*(1 + epsilon**2))
    dy2dt_mj = lambda q,R0: -q[1]/(2*np.exp(q[3])) - (np.exp(q[3])*R0*(-1 + q[1])*(1 + epsilon)*(q[0] + q[1] - q[0]*epsilon
                          + q[1]*epsilon))/(4*(1 + epsilon**2))
    dp1dt_mj = lambda q,R0: (-(np.exp(2*q[2])*R0*(-1 + epsilon)*(-1 + q[1] - 2*q[0]*(-1 + epsilon) + epsilon + q[1]*epsilon))
                 - np.exp(q[2] + q[3])*R0*(-1 + q[1])*(-1 + epsilon**2) - 2*(1 + epsilon**2) -2*np.exp(q[2])*(-1 -
                epsilon**2 +R0*(-1 + epsilon)*(1 + q[0]*(-1 + epsilon) - q[1]*(1 + epsilon))))/(4*np.exp(q[2])*(1 + epsilon**2))
    dp2dt_mj = lambda q,R0: (-(np.exp(q[2] + q[3])*R0*(-1 + q[0])*(-1 + epsilon**2)) - 2*(1 + epsilon**2) +
                    np.exp(2*q[3])*R0*(1 + epsilon)*(q[0] - q[0]*epsilon + (-1 + 2*q[1])*(1 + epsilon)) +2*np.exp(q[3])*
                    (1 + epsilon**2+ R0*(1 + epsilon)*(1 + q[0]*(-1 + epsilon) - q[1]*(1 + epsilon))))/(4*np.exp(q[3])*(1 + epsilon**2))
    dq_dt = lambda q,t:np.array([dy1dt_mj(q, xi * lam if t > t0 and t < tf else lam),dy2dt_mj(q,xi*lam if t>t0 and t<tf else lam),
                                 dp1dt_mj(q, xi * lam if t > t0 and t < tf else lam),dp2dt_mj(q, xi * lam if t > t0 and t < tf else lam)])
    y1_0, y2_0, p1_0, p2_0, p1_star, p2_star = eq_points_mj(epsilon,lam)
    return y1_0, y2_0, p1_0, p2_0, p1_star, p2_star, dq_dt, ndft.Jacobian(dq_dt)

def postive_eigen_vec(J,q0):
    # Find eigen vectors
    eigen_value, eigen_vec = la.eig(J(q0,0.0))
    postive_eig_vec = []
    for e in range(np.size(eigen_value)):
        if eigen_value[e].real > 0:
            postive_eig_vec.append(eigen_vec[:, e].reshape(4, 1).real)
    return postive_eig_vec


def shoot(y1_0, y2_0, p1_0, p2_0, tshot, J,dq_dt,lam,epsilon,factor,gamma,qstar):
    q0 = (y1_0, y2_0, p1_0, p2_0)
    vect_J = lambda q, tshot: J(q0,0.0)
    # qsol = odeint(dq_dt, q0, tshot,atol=1.0e-20, rtol=1.0e-13, mxstep=1000000000, hmin=1e-30, Dfun=vect_J)
    # qsol = odeint(dq_dt, q0, tshot,args=(lam,epsilon,factor,gamma,duration,t0),Dfun=vect_J,tcrit=qstar)
    qsol = odeint(dq_dt, q0, tshot,Dfun=vect_J,tcrit=qstar)
    return qsol


def one_shot(shot_angle,lin_weight,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt,epsilon,lam,factor,gamma,duration,t0):
    q0 = (q_star[0] + radius * np.cos(shot_angle), q_star[1], 0+radius * np.sin(shot_angle), 0)
    postive_eig_vec = postive_eigen_vec(J, q0)
    y1_i, y2_i, p1_i, p2_i = q0[0] + lin_weight * float(postive_eig_vec[0][0]) * one_shot_dt + (
                1 - lin_weight) * float(postive_eig_vec[1][0]) * one_shot_dt \
        , q0[1] + float(lin_weight * postive_eig_vec[0][1]) * one_shot_dt + (1 - lin_weight) * float(
        postive_eig_vec[1][1]) * one_shot_dt \
        , q0[2] + float(postive_eig_vec[0][2]) * one_shot_dt + (1 - lin_weight) * float(postive_eig_vec[1][2]) * one_shot_dt \
        , q0[3] + lin_weight * float(postive_eig_vec[0][3]) * one_shot_dt + (1 - lin_weight) * float(
        postive_eig_vec[1][3]) * one_shot_dt
    # y1_i, y2_i, p1_i, p2_i =  q0[0] + lin_weight * float(postive_eig_vec[0][0]) * one_shot_dt + (
    #             1 - lin_weight) * float(postive_eig_vec[1][0]) * one_shot_dt \
    #     , q0[1] + float(lin_weight * postive_eig_vec[0][1]) * one_shot_dt + (1 - lin_weight) * float(
    #     postive_eig_vec[1][1]) * one_shot_dt \
    #     , q0[2] + float(postive_eig_vec[0][2]) * one_shot_dt + (1 - lin_weight) * float(postive_eig_vec[1][2]) * one_shot_dt \
    #     , -2.1e-12+q0[3] + lin_weight * float(postive_eig_vec[0][3]) * one_shot_dt + (1 - lin_weight) * float(
    #     postive_eig_vec[1][3]) * one_shot_dt
    return shoot(y1_i, y2_i, p1_i, p2_i, final_time_path,J,shot_dq_dt,lam,epsilon,factor,gamma,[0.0,final_time_path[-1]])

def record_data(folder_name,beta,gamma,stoptime,int_lin_combo,numpoints,epsilon,path,q_star,r,angle,factor,t0,duration):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path + '/Data')
    os.mkdir(folder_name)
    os.chdir(dir_path + '/Data/' + folder_name)
    pickle.dump(beta, open('beta.pkl', 'wb'))
    pickle.dump(gamma, open('gamma.pkl', 'wb'))
    # pickle.dump(sim, open('sim.pkl', 'wb'))
    pickle.dump(stoptime, open('stoptime.pkl', 'wb'))
    pickle.dump(int_lin_combo, open('lin_combo.pkl', 'wb'))
    pickle.dump(numpoints, open('numpoints.pkl', 'wb'))
    pickle.dump(epsilon, open('epsilon_matrix.pkl', 'wb'))
    pickle.dump(path, open('guessed_paths.pkl', 'wb'))
    pickle.dump(np.linspace(0.0, stoptime, numpoints), open('time_series.pkl', 'wb'))
    # pickle.dump(guessed_action, open('action_paths.pkl', 'wb'))
    pickle.dump(q_star, open('qstar.pkl', 'wb'))
    pickle.dump(r, open('radius.pkl', 'wb'))
    pickle.dump(angle, open('shot_angle.pkl', 'wb'))
    pickle.dump(factor, open('factor.pkl', 'wb'))
    pickle.dump(t0, open('t0.pkl', 'wb'))




if __name__=='__main__':
    #Network Parameters
    beta, gamma = 1.3, 1.0
    # epslam,epsmu = 0.4,-0.4
    epsilon = 0.000000001

    lam = beta/(1+epsilon**2)
    factor = 0.75
    abserr,relerr = 1.0e-20,1.0e-13
    # ODE parameters
    stoptime = 12.5
    numpoints = 10000
    # Create the time samples for the output of the ODE solver
    t = np.linspace(0.0,stoptime,numpoints)
    dt = 1e-3
    # Radius around eq point, shooting angle and linear combination of the the eigenvectors
    r = 9.345e-4
    angle = np.pi*1.0
    duration,t0 = 5.0,3.0
    tf = t0 + duration
    folder_name='epslam01_epsin01_factor0_lam16_stoptime20_t001_angle10_duration01_numpoints2000'
    int_lin_combo = 1.0 # cat 0.0 epslam = 0.1 epsmu=0.1 t0=1.0 duration=2.0
    y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, shot_dq_dt, J = eq_motion_mj(lam,t0,tf, factor,epsilon)
    q_star = [y1_0, y2_0, p1_star_clancy, p2_star_clancy]
    path = one_shot(angle, int_lin_combo, q_star, r, t, dt, J, shot_dq_dt,epsilon,lam,factor,gamma,duration,t0)
    # y1_for_linear = np.linspace(path[:, 0][-1], 0, numpoints)
    # py1_linear = p1_star_clancy - ((p1_star_clancy - path[:, 2][-1]) / path[:, 0][-1]) * y1_for_linear
    # y2_for_linear = np.linspace(path[:, 1][-1], 0, numpoints)
    # py2_linear = p2_star_clancy - ((p2_star_clancy - path[:, 3][-1]) / path[:, 1][-1]) * y2_for_linear
    # addition_to_path = np.stack((y1_for_linear, y2_for_linear, py1_linear, py2_linear), axis=1)
    # path_addition = np.vstack((path, addition_to_path))
    # plt.plot((path_addition[:,0]+path_addition[:,1])/2,(path_addition[:,2]+path_addition[:,3])/2)
    # plt.plot((path[:,0]+path[:,1])/2,(path[:,2]+path[:,3])/2)
    # plt.xlabel('w')
    # plt.ylabel(r'$p_{w}$')
    # plt.title(r'$p_{w}$ vs w for $R_{0}$='+str(beta))
    # plt.show()
    # plt.plot(path[:,0],path[:,2],label='y1')
    # plt.xlabel(r'$y_{1}$')
    # plt.ylabel(r'$p_{1}$')
    # plt.title(r'$p_{1}$ vs $y_{1}$ for $R_{0}$='+str(beta))
    # plt.show()
    # plt.plot(path[:,1],path[:,3],label='y2')
    plt.plot(t,path[:,0],label='y2')
    plt.scatter(0.0,y1_0)
    plt.scatter(stoptime,0.0)
    # plt.xlabel(r'$y_{2}$')
    # plt.ylabel(r'$p_{2}$')
    plt.title(r'$p_{2}$ vs $y_{2}$ for $R_{0}$='+str(beta))
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p$')
    plt.legend()
    plt.title(r'$p$ vs $y$ for $R_{0}$='+str(beta))

    plt.show()


    # plt.plot(t, (path[:,0]+path[:,1])/2)
    # plt.xlabel('t')
    # plt.ylabel(r'$w$')
    # plt.title(r'w vs t for $R_{0}$=' + str(beta))
    # plt.show()
    # record_data(folder_name, beta, gamma, stoptime, int_lin_combo, numpoints, epsilon, path, q_star, r, angle,factor,t0,duration)
    print('This no love song')