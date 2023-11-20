# This program solves the Hamilton equations for a bimodal networks with heterogeneous rates SIS dynamics.
# The resulting action and paths are stored in a pickle, where they can be accessed

import numpy as np
from scipy.integrate import odeint
import numdifftools as ndft
import os
import plot1d
import pickle

def eq_points_1d(R0):
    return (1-1/R0),0.0,-np.log(R0)
    # return (1/2)*(1-1/R0),0.0,-2*np.log(R0)

def eq_motion_1d(lam,t0,tf,xi):
    # Hamilton eq for 1d case
    # dy_dt_sus_inf = lambda q, R0:-((q[0]*(1 + np.exp(q[1])*R0*(-1 + 2*q[0])))/np.exp(q[1]/2))
    dy_dt_sus_inf = lambda q, R0:-(((1 + np.exp(2*q[1])*R0*(-1 + q[0]))*q[0])/np.exp(q[1]))
    # dtheta_dt = lambda q, R0: (2 * (-1 + np.exp(q[1] / 2)) * (1 + np.exp(q[1] / 2) * R0 * (-1 + 4 * q[0])))/np.exp(q[1]/2)
    dtheta_dt = lambda q, R0: ((-1 + np.exp(q[1]))*(1 + np.exp(q[1])*R0*(-1 + 2*q[0])))/np.exp(q[1])
    dq_dt_sus_inf = lambda q, t=None: np.array([dy_dt_sus_inf(q, xi * lam if t > t0 and t < tf else lam),
                                                dtheta_dt(q,xi*lam if t>t0 and t<tf else lam)])
    y0, p0, pstar = eq_points_1d(lam)
    return y0, p0, pstar, dq_dt_sus_inf, ndft.Jacobian(dq_dt_sus_inf)

def multi_eps_normalized_path(beta,numpoints,radius,tf,xi,t0,endtimesim,abserr,relerr):
    y0,p0, pstar,dq_dt,J = eq_motion_1d(beta,t0,tf,xi)
    q_star = [y0,pstar]
    q0 = (q_star[0] - radius, radius*q_star[1]/q_star[0])
    vect_J = lambda q, tshot: J(q0)
    t = np.linspace(0.0,endtimesim,numpoints)
    qsol = odeint(dq_dt, q0, t)
    return qsol,q_star


def record_data(folder_name,beta,stoptime,numpoints,guessed_paths,qstar,xi,t0,tf,radius):
    # Export the simulation data to pickle files
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path+'/Data1d')
    os.mkdir(folder_name)
    os.chdir(dir_path+'/Data1d/'+folder_name)
    pickle.dump(beta,open('beta.pkl','wb'))
    pickle.dump(stoptime,open('stoptime.pkl','wb'))
    pickle.dump(numpoints,open('numpoints.pkl','wb'))
    pickle.dump(guessed_paths,open('guessed_paths.pkl','wb'))
    pickle.dump(np.linspace(0.0, stoptime, numpoints),open('time_series.pkl','wb'))
    pickle.dump(qstar,open('qstar.pkl','wb'))
    pickle.dump(xi,open('xi.pkl','wb'))
    pickle.dump(t0,open('t0.pkl','wb'))
    pickle.dump(tf,open('tf.pkl','wb'))
    pickle.dump(radius,open('radius.pkl','wb'))


if __name__=='__main__':
    # ODE parameters
    abserr,relerr = 1.0e-16,1.0e-13
    # Network Parameters
    beta, gamma = 1.3, 1.0
    stoptime = 40.0
    numpoints = 10000
    # t0,tf,xi = 17.0,19.0,0.9
    t0,tf,xi = 17.0,19.0,0.875
    # Create the time samples for the output of the ODE solver
    t = np.linspace(0.0,stoptime,numpoints)
    # Radius around eq point, shooting angle and linear combination of the the eigenvectors
    # r = 2.773e-8
    # r = 4.27816e-8
    # r = 6.742e-8
    # r = 6.742e-8
    # r = 7.055e-8

    # r = 9.8e-4
    # r = 8.915e-4
    # r = 1.078e-3
    # r = 1.0273e-3
    r = 9.345e-4

    # r = 2.7879e-8

    path,qstar = multi_eps_normalized_path(beta,numpoints,r,tf,xi,t0,stoptime,abserr,relerr)

    directory_name = '/Data1d/'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    plot1d.plot_paths_energy(stoptime, beta, path, xi, tf-t0, dir_path, directory_name,tf-t0,t)

    folder_name='path_R0_{}_stoptime_{}_t0_{}_tf_{}_xi_{}_rad_{}'.format(beta,stoptime,t0,tf,xi,r)
    record_data(folder_name,beta,stoptime,numpoints,path,qstar,xi,t0,tf,r)
