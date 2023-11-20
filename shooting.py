# This program solves the Hamilton equations for a bimodal networks with heterogeneous rates SIS dynamics.
# The resulting action and paths are stored in a pickle, where they can be accessed

import numpy as np
import scipy.linalg as la
from scipy.integrate import odeint
from scipy.integrate import simps
import numdifftools as ndft
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

def eq_points_undirected(epsilon,R0):
    # for a 2d bimodal undirected network the fixed points are
    y1star =(R0 + 2 * (-1 + R0) *epsilon + 3 * R0 * epsilon ** 2 + 2 * (-1 + R0) *epsilon ** 3 -2 * np.sqrt(
            (1 - R0) / (-1 + epsilon ** 2) +(1 / (-1 + epsilon ** 2) + R0 / (2 + 2 *epsilon ** 2)) ** 2) +2 *epsilon ** 4 *
            np.sqrt((1 - R0) / (-1 + epsilon ** 2) +(1 / (-1 + epsilon ** 2) + R0 / (2 + 2 *epsilon ** 2)) ** 2)) /\
            (4 * R0 *epsilon * (1 + epsilon) ** 2)
    y2star =(R0 + 2*(-1 + R0)*epsilon + 3*R0*epsilon**2 + 2*(-1 + R0)*epsilon**3 - 2*np.sqrt((1 - R0)/(-1 + epsilon**2) +
            (1/(-1 + epsilon**2) + R0/(2 + 2*epsilon**2))**2) + 2*epsilon**4*np.sqrt((1 - R0)/(-1 + epsilon**2) +
            (1/(-1 + epsilon**2) + R0/(2 + 2*epsilon**2))**2))/(4*R0*epsilon*(1 + epsilon)**2)
    p1star =-np.log(1 + (1 - epsilon)*(1/(-1 + epsilon**2) + R0/(2 + 2*epsilon**2) + np.sqrt((1 - R0)/(-1 + epsilon**2) +
            (1/(-1 + epsilon**2) + R0/(2 + 2*epsilon**2))**2)))
    p2star =-np.log(1 + (1 + epsilon)*(1/(-1 + epsilon**2) + R0/(2 + 2*epsilon**2) + np.sqrt((1 - R0)/(-1 + epsilon**2) +
            (1/(-1 + epsilon**2) + R0/(2 + 2*epsilon**2))**2)))
    return y1star, y2star, 0.0, 0.0, p1star, p2star


def eq_hamilton_J(beta,epsilon=0.0,t=None,gamma=1.0):
    # find the end and starting point along the path
    def bimodal_mu_lam(lam=beta):
        epsilon_lam,epsilon_mu=epsilon[0],epsilon[1]
        dy1_dt_sus_inf = lambda q: lam * ((1 - epsilon_lam) * q[0] + (1 + epsilon_lam) * q[1]) * (1 - epsilon_mu) * (
                    1 / 2 - q[0]) * np.exp(q[2]) - gamma * q[0] * np.exp(-q[2])
        dy2_dt_sus_inf = lambda q: lam * ((1 - epsilon_lam) * q[0] + (1 + epsilon_lam) * q[1]) * (1 + epsilon_mu) * (
                    1 / 2 - q[1]) * np.exp(q[3]) - gamma * q[1] * np.exp(-q[3])
        dtheta1_dt_sus_inf = lambda q: -lam * (1 - epsilon_lam) * ((1 - epsilon_mu) * (1 / 2 - q[0]) * (np.exp(q[2]) - 1)
                    + (1 + epsilon_mu) * (1 / 2 - q[1]) * (np.exp(q[3]) - 1)) + lam * ((1 - epsilon_lam) * q[0]
                    + (1 + epsilon_lam) * q[1]) * (1 - epsilon_mu) * (np.exp(q[2]) - 1) - gamma * (np.exp(-q[2]) - 1)
        dtheta2_dt_sus_inf = lambda q: -lam * (1 + epsilon_lam) * ((1 - epsilon_mu) * (1 / 2 - q[0]) * (np.exp(q[2]) - 1)
                    + (1 + epsilon_mu) * (1 / 2 - q[1]) * (np.exp(q[3]) - 1)) + lam * ((1 - epsilon_lam) * q[0] +
                    (1 + epsilon_lam) * q[1]) * (1 + epsilon_mu) * (np.exp(q[3]) - 1) - gamma * (np.exp(-q[3]) - 1)
        dq_dt_sus_inf = lambda q, t=None: np.array(
            [dy1_dt_sus_inf(q), dy2_dt_sus_inf(q), dtheta1_dt_sus_inf(q), dtheta2_dt_sus_inf(q)])
        return dq_dt_sus_inf

def eq_motion_undirected(lam,epsilon=0.0,t0=0.0,tf=10.0,xi=1.0,gamma=1.0):
    def hamilon_eqautions(lam):
        # Hamilton's equations for undirected bimodal network
        dy1_dt_sus_inf = lambda q,R0:-((q[0]*gamma)/np.exp(q[2])) + np.exp(q[2])*R0*(0.5 - q[0])*(1 - epsilon)*(q[0] + q[1] - q[0]*epsilon + q[1]*epsilon)
        dy2_dt_sus_inf = lambda q,R0:-((q[1]*gamma)/np.exp(q[3])) - (np.exp(q[3])*R0*(-1 + 2*q[1])*(1 + epsilon)*(q[0] + q[1] - q[0]*epsilon + q[1]*epsilon))/2
        dtheta1_dt_sus_inf = lambda q,R0:gamma - gamma/np.exp(q[2]) + (R0*(-1 + epsilon)*(-2 - 4*q[0]*(-1 + epsilon) + 4*q[1]*(1 + epsilon) -
               np.exp(q[3])*(-1 + 2*q[1])*(1 + epsilon) + np.exp(q[2])*(1 + 4*q[0]*(-1 + epsilon) - epsilon - 2*q[1]*(1 + epsilon))))/2
        dtheta2_dt_sus_inf = lambda q,R0:gamma - gamma/np.exp(q[3]) - (R0*(1 + epsilon)*(-2 - 4*q[0]*(-1 + epsilon) + np.exp(q[2])*(-1 + 2*q[0])*(-1 + epsilon) +
                4*q[1]*(1 + epsilon) + np.exp(q[3])*(1 - 4*q[1] + 2*q[0]*(-1 + epsilon) + epsilon - 4*q[1]*epsilon)))/2
        dq_dt_sus_inf = lambda q, t=None: np.array(
            [dy1_dt_sus_inf(q,xi*lam if t>t0 and t<tf else lam), dy2_dt_sus_inf(q,xi*lam if t>t0 and t<tf else lam), dtheta1_dt_sus_inf(q), dtheta2_dt_sus_inf(q)])
        return dq_dt_sus_inf

    # Hamilton eq for 2d case
    dq_dt_sus_inf = hamilon_eqautions(lam/(1+epsilon**2))
    y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy = eq_points_undirected(epsilon, lam/(1+epsilon**2))
    return y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, dq_dt_sus_inf, ndft.Jacobian(dq_dt_sus_inf)


def postive_eigen_vec(J,q0):
    # Find eigen vectors
    eigen_value, eigen_vec = la.eig(J(q0,None))
    postive_eig_vec = []
    for e in range(np.size(eigen_value)):
        if eigen_value[e].real > 0:
            postive_eig_vec.append(eigen_vec[:, e].reshape(4, 1).real)
    return postive_eig_vec


def shoot(y1_0, y2_0, p1_0, p2_0, tshot, J,dq_dt):
    q0 = (y1_0, y2_0, p1_0, p2_0)
    vect_J = lambda q, tshot: J(q0)
    qsol = odeint(dq_dt, q0, tshot,atol=1.0e-20, rtol=1.0e-13, mxstep=1000000000, hmin=1e-30, Dfun=vect_J)
    return qsol


def one_shot(shot_angle,lin_weight,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt):
    q0 = (q_star[0] + radius * np.cos(shot_angle), q_star[1], 0+radius * np.sin(shot_angle), 0)
    postive_eig_vec = postive_eigen_vec(J, q0)
    y1_i, y2_i, p1_i, p2_i = q0[0] + lin_weight * float(postive_eig_vec[0][0]) * one_shot_dt + (
                1 - lin_weight) * float(postive_eig_vec[1][0]) * one_shot_dt \
        , q0[1] + float(lin_weight * postive_eig_vec[0][1]) * one_shot_dt + (1 - lin_weight) * float(
        postive_eig_vec[1][1]) * one_shot_dt \
        , q0[2] + float(postive_eig_vec[0][2]) * one_shot_dt + (1 - lin_weight) * float(postive_eig_vec[1][2]) * one_shot_dt \
        , q0[3] + lin_weight * float(postive_eig_vec[0][3]) * one_shot_dt + (1 - lin_weight) * float(
        postive_eig_vec[1][3]) * one_shot_dt
    return shoot(y1_i, y2_i, p1_i, p2_i, final_time_path,J,shot_dq_dt)

def path_diverge(path):
    if path[:,2][np.absolute(path[:,2])>=10.0].size is not 0 or path[:,3][np.absolute(path[:,3])>=10.0].size is not 0:return True
    return False

def when_path_diverge(path):
    p1_max_div = np.where(np.absolute(path[:, 2]) > 10.0)
    p1_div = 0.0 if not len(p1_max_div[0]) else np.where(np.absolute(path[:, 2]) > 10.0)[0][0]
    p2_max_div = np.where(np.absolute(path[:, 3]) > 10.0)
    p2_div = 0.0 if not len(p2_max_div[0]) else np.where(np.absolute(path[:, 3]) > 10.0)[0][0]
    if float(p2_div)==float(p1_div)==0.0:return 0.0
    if p1_div is 0.0: return p2_div
    if p2_div is 0.0: return p1_div
    return min(p1_div,p2_div)


def change_shot_angle(shot_angle,radius,org_lin_combo,one_shot_dt,q_star,final_time_path,J,shot_dq_dt):
    path = one_shot(shot_angle,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
    going_up = path[:,0][-10]+path[:,1][-10]>=path[:,0][0]+path[:,1][0]
    dtheta=1e-3
    shot_angle_down,shot_angle_up=shot_angle - dtheta,shot_angle + dtheta
    count,max_steps=0,int(np.pi/dtheta)
    while going_up and max_steps>count:
        path_down=one_shot(shot_angle_down,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
        if not path_down[:, 0][-10] + path_down[:, 1][-10] >= path_down[:, 0][0] + path_down[:, 1][0]:
            shot_angle=shot_angle_down
            break
        path_up=one_shot(shot_angle_up,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
        if not path_up[:, 0][-10] + path_up[:, 1][-10] >= path_up[:, 1][0] + path_up[:, 0][0]:
            shot_angle=shot_angle_up
            break
        shot_angle_down, shot_angle_up = shot_angle_down - dtheta, shot_angle_up + dtheta
        count=count+1
    return shot_angle


def path_going_up(shot_angle,radius,org_lin_combo,one_shot_dt,q_star,final_time_path,J,shot_dq_dt):
    org_radius=radius
    path=one_shot(shot_angle,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
    org_div_time = when_path_diverge(path)
    going_up = path[:,0][int(org_div_time)-2]+path[:,1][int(org_div_time)-2]>=path[:,0][0]+path[:,1][0]
    while going_up:
        radius=radius*2
        path=one_shot(shot_angle,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
        org_div_time = when_path_diverge(path)
        going_up = path[:,0][int(org_div_time)-2]+path[:,1][int(org_div_time)-2]>=path[:,0][0]+path[:,1][0]
    if radius>5e-3:
        shot_angle=change_shot_angle(shot_angle,org_radius,org_lin_combo,one_shot_dt,q_star,final_time_path,J,shot_dq_dt)
        radius=org_radius
    path=one_shot(shot_angle,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
    return path,radius,shot_angle


def best_diverge_path(shot_angle,radius,org_lin_combo,one_shot_dt,q_star,final_time_path,J,shot_dq_dt):
    path,radius,shot_angle=path_going_up(shot_angle,radius,org_lin_combo,one_shot_dt,q_star,final_time_path,J,shot_dq_dt)
    org_div_time = when_path_diverge(path)
    dl,lin_combo = 0.1,org_lin_combo
    while path_diverge(path) is True:
        org_div_time = when_path_diverge(path)
        lin_combo_step_up=lin_combo+dl
        lin_combo_step_down=lin_combo-dl
        path_up=one_shot(shot_angle,lin_combo_step_up,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
        path_down=one_shot(shot_angle,lin_combo_step_down,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
        time_div_up,time_div_down=when_path_diverge(path_up),when_path_diverge(path_down)
        if time_div_down == 0.0:
            return lin_combo_step_down,radius,shot_angle
        if time_div_up == 0.0:
            return lin_combo_step_up,radius,shot_angle
        best_time_before_diverge=max(time_div_up,time_div_down,org_div_time)
        if best_time_before_diverge == org_div_time:
            dl=dl/10
        elif best_time_before_diverge is time_div_down:
            lin_combo= lin_combo-dl
        elif best_time_before_diverge is time_div_up:
            lin_combo=lin_combo+dl
        path = one_shot(shot_angle,lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
    return lin_combo,radius,shot_angle

def fine_tuning(shot_angle,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt):
    # make sure that the paths is at least close to the end point by min_accurecy
    min_accurecy,dl=1e-6,1e-4
    path = one_shot(shot_angle,org_lin_combo,q_star,radius,final_time_path,one_shot_dt,J,shot_dq_dt)
    distance_from_theory = lambda p:(p[:,0][-1]+p[:,1][-1])/2
    lin_combo = org_lin_combo
    current_distance = distance_from_theory(path)
    if path[:,0][-1]+path[:,1][-1]<1e-1:
        while current_distance>min_accurecy:
            lin_combo_step_up = lin_combo + dl
            lin_combo_step_down = lin_combo - dl
            path_up = one_shot(shot_angle, lin_combo_step_up, q_star,radius, final_time_path,one_shot_dt,J,shot_dq_dt)
            path_down = one_shot(shot_angle, lin_combo_step_down,q_star ,radius, final_time_path, one_shot_dt,J,shot_dq_dt)
            if not path_diverge(path_up) and distance_from_theory(path_up)<current_distance:
                current_distance=distance_from_theory(path_up)
                lin_combo=lin_combo+dl
            elif not path_diverge(path_down) and distance_from_theory(path_down)<current_distance:
                current_distance=distance_from_theory(path_down)
                lin_combo=lin_combo-dl
            elif dl<1e-16:
                break
            else:
                dl=dl/2
    return lin_combo

def guess_path(sampleingtime,shot_angle,lin_combo,q_star,one_shot_dt,org_radius,sample_size,J,shot_dq_dt):
    # find the path with a given parameters
    radius=org_radius
    for s in sampleingtime:
        lin_combo, radius,shot_angle = best_diverge_path(shot_angle, radius,lin_combo,one_shot_dt,q_star,np.linspace(0.0,s,sample_size),J,shot_dq_dt )
    lin_combo, radius,shot_angle = best_diverge_path(shot_angle, org_radius, lin_combo, one_shot_dt, q_star,np.linspace(0.0, sampleingtime[-1], sample_size), J, shot_dq_dt)
    # lin_combo= fine_tuning(shot_angle, lin_combo,q_star,radius, np.linspace(0.0,sampleingtime[-1],sample_size), one_shot_dt,J,shot_dq_dt)
    return lin_combo,radius,shot_angle, one_shot(shot_angle, lin_combo,q_star,radius,np.linspace(0.0,sampleingtime[-1],sample_size),one_shot_dt,J,shot_dq_dt)

def multi_eps_normalized_path(case_to_run,list_of_epsilons,beta,gamma,numpoints,one_shot_dt,radius,
                              lin_combo=1.00008204478397,org_shot_angle=np.pi/4-0.785084,action_times=None,tf=100,xi=1.0):
    # A function that recives paths parameters, with either different epsilon or different reproduction number
    # and run simulation for each one
    guessed_paths,guessed_lin_combo,guessed_qstar,guessed_action,guessed_r,guessed_angle,guess_action_time_series,guessed_action_part=[],[],[],[],[],[],[],[]
    shot_angle=org_shot_angle
    if type(beta) is list:
        # if the reproduction number is differ between simulations
        for l in beta:
            if l<=1.8:
                sampleingtime=[30.0]

            elif l<=2.4:
                sampleingtime = [15.0]
            elif l<=3.3:
                sampleingtime = [10.0]
            elif l<=4.4:
                sampleingtime = [7.0]
            else:
                sampleingtime = [5.0]
            y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, shot_dq_dt, J = eq_motion_undirected(case_to_run, l,
                                                                                                  list_of_epsilons, t, gamma)
            q_star = [y1_0, y2_0, p1_star_clancy, p2_star_clancy]
            lin_combo,temp_radius,shot_angle,path=guess_path(sampleingtime,shot_angle,lin_combo,q_star,one_shot_dt,radius,numpoints,J,shot_dq_dt)
            guessed_paths.append(path)

            guessed_lin_combo.append(lin_combo)
            guessed_qstar.append(q_star)
            guessed_action.append(simulation_action(path,q_star))
            guessed_r.append(temp_radius)
            guessed_angle.append(shot_angle)

    else:
        # if each simulation have different epsilon
        for eps in list_of_epsilons:
            sampleingtime=[20.0]
            y1_0, y2_0, p1_0, p2_0, p1_star_clancy, p2_star_clancy, shot_dq_dt,J = eq_motion_undirected(beta,eps,t,tf,xi)
            q_star = [y1_0, y2_0, p1_star_clancy, p2_star_clancy]
            lin_combo,temp_radius,shot_angle,path = guess_path(sampleingtime,shot_angle,lin_combo,q_star,one_shot_dt,radius,numpoints,J,shot_dq_dt)
            if action_times is not None:
                path_time,path_act=[],[]
                for time in action_times:
                    path_current_eps=one_shot(shot_angle, lin_combo, q_star, temp_radius, np.linspace(0.0, time, numpoints), one_shot_dt, J,shot_dq_dt)
                    path_time.append(path_current_eps)
                    path_act.append(simps(path_current_eps[:, 2], path_current_eps[:, 0]) + simps(path_current_eps[:, 3], path_current_eps[:, 1]))
            guessed_paths.append(path)
            guessed_lin_combo.append(lin_combo)
            guessed_qstar.append(q_star)
            guessed_action.append(simulation_action(path,q_star))
            guessed_r.append(temp_radius)
            guessed_angle.append(shot_angle)
    return guessed_paths,sampleingtime[-1],guessed_lin_combo,guessed_qstar,guessed_action,guessed_r,guessed_angle


def simulation_action(path,q_star):
    # Find the action along the path
    y1_for_linear,y2_for_linear = np.linspace(path[:, 0][-1], 0, 1000),np.linspace(path[:, 1][-1], 0, 1000)
    py1_linear = q_star[2] - ((q_star[2] - path[:, 2][-1]) / path[:, 0][-1]) * y1_for_linear
    py2_linear = q_star[3] - ((q_star[3] - path[:, 3][-1]) / path[:, 1][-1]) * y2_for_linear
    return simps(path[:, 2], path[:, 0]) + simps(path[:, 3], path[:, 1]) + simps(py1_linear, y1_for_linear) + simps(py2_linear, y2_for_linear)


def record_data(folder_name,beta,gamma,sim,stoptime,int_lin_combo,numpoints,epsilon_matrix,guessed_paths,guessed_action,qstar,rad,ang):
    # Export the simulation data to pickle files
    dir_path= os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path+'/Data')
    os.mkdir(folder_name)
    os.chdir(dir_path+'/Data/'+folder_name)
    pickle.dump(beta,open('beta.pkl','wb'))
    pickle.dump(gamma,open('gamma.pkl','wb'))
    pickle.dump(sim,open('sim.pkl','wb'))
    pickle.dump(stoptime,open('stoptime.pkl','wb'))
    pickle.dump(int_lin_combo,open('lin_combo.pkl','wb'))
    pickle.dump(numpoints,open('numpoints.pkl','wb'))
    pickle.dump(epsilon_matrix,open('epsilon_matrix.pkl','wb'))
    pickle.dump(guessed_paths,open('guessed_paths.pkl','wb'))
    pickle.dump(np.linspace(0.0, stoptime, numpoints),open('time_series.pkl','wb'))
    pickle.dump(guessed_action,open('action_paths.pkl','wb'))
    pickle.dump(qstar,open('qstar.pkl','wb'))
    pickle.dump(rad,open('radius.pkl','wb'))
    pickle.dump(ang,open('shot_angle.pkl','wb'))


if __name__=='__main__':
    #Network Parameters
    beta, gamma = 1.6, 1.0
    abserr,relerr = 1.0e-20,1.0e-13
    sim='x'
    # ODE parameters
    stoptime=20.0
    numpoints = 10000
    # Create the time samples for the output of the ODE solver
    t = np.linspace(0.0,stoptime,numpoints)
    dt = 1e-3
    # Radius around eq point, shooting angle and linear combination of the the eigenvectors
    r = 5e-10
    angle = 2.510116743320488
    int_lin_combo = 1.0035317699739623

    # epsilon_matrix = [[(0.3,e) for e in np.linspace(-0.3,0.3,10)]]
    epsilon_matrix = [[e for e in np.linspace(-0.3,0.3,10)]]
    sim_paths,sim_sampletime,sim_lin_combo,sim_action,sim_qstar,sim_r,sim_angle=[],[],[],[],[],[],[]

    for case,epsilons in zip(sim,epsilon_matrix):
        path,sampletime,lin_combo,qstar,path_action,rad,ang=multi_eps_normalized_path(case, epsilons, beta, gamma, numpoints, dt, r, int_lin_combo,angle)
        sim_paths.append(path)
        sim_sampletime.append(sampletime)
        sim_lin_combo.append(lin_combo)
        sim_action.append(path_action)
        sim_qstar.append(qstar)
        sim_r.append(rad)
        sim_angle.append(ang)
    folder_name='data'
    record_data(folder_name,beta,gamma,sim,sim_sampletime,sim_lin_combo,numpoints,epsilon_matrix,sim_paths,sim_action,sim_qstar,sim_r,sim_angle)