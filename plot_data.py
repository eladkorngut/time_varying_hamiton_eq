import numpy as np
import scipy.linalg as la
from scipy.integrate import simps
import matplotlib.pyplot as plt
import pickle
import os

hamiltonian = lambda R0,y1,y2,p1,p2,eps_mu,eps_lam: (-1 + np.exp(-p1))*y1 + (-1 + np.exp(-p2))*y2 + (R0*(y1 + y2 + (-y1 + y2)*
              eps_lam)*(((-1 + np.exp(p1))*(-1 + 2*y1)*(-1 + eps_mu))/2 + (-1 + np.exp(p2))*(0.5 - y2)*(1 + eps_mu)))/(1 + eps_lam*eps_mu)

def plot_paths_energy(t,lam,p,e,f,dir_path,directory_name):
    fig_time_v_w, ax_time_v_w = plt.subplots()
    fig_time_v_y1, ax_time_v_y1 = plt.subplots()
    fig_time_v_y2, ax_time_v_y2 = plt.subplots()
    fig_time_v_u, ax_time_v_u = plt.subplots()
    fig_time_h, ax_time_h = plt.subplots()
    fig_y1_p1, ax_y1_p1 = plt.subplots()
    fig_y2_p2, ax_y2_p2 = plt.subplots()
    fig_w_pw, ax_w_pw = plt.subplots()
    fig_u_pu, ax_u_pu = plt.subplots()
    linstyles=['-','--',':','-.']
    for i,(time_series,beta,path,epsilon,factor) in enumerate(zip(t,lam,p,e,f)):
        w,u,pw,pu = (path[:, 0]+path[:, 1])/2,(path[:, 0]-path[:, 1])/2,(path[:, 2] + path[:, 3]) / 2,(path[:, 2] - path[:, 3]) / 2

        ax_time_h.plot(time_series,hamiltonian(beta, path[:, 0], path[:, 1], path[:, 2], path[:, 3], epsilon[0],
                             epsilon[1]),linestyle=linstyles[i%len(linstyles)],label=r'$\phi={}$'.format(factor))
        ax_time_v_w.plot(time_series,w,linestyle=linstyles[i%len(linstyles)],label=r'$\phi={}$'.format(factor))
        ax_time_v_u.plot(time_series,u,linestyle=linstyles[i%len(linstyles)],label=r'$\phi={}$'.format(factor))
        ax_time_v_y1.plot(time_series,path[:, 0],linestyle=linstyles[i%len(linstyles)],label=r'$\phi={}$'.format(factor))
        ax_time_v_y2.plot(time_series,path[:, 1],linestyle=linstyles[i%len(linstyles)],label=r'$\phi={}$'.format(factor))
        ax_y1_p1.plot(path[:, 0],path[:, 2],linestyle=linstyles[i%len(linstyles)],label=r'$\phi={}$'.format(factor))
        ax_y2_p2.plot(path[:, 1],path[:, 3],linestyle=linstyles[i%len(linstyles)],label=r'$\phi={}$'.format(factor))
        ax_u_pu.plot(u,pu,linestyle=linstyles[i%len(linstyles)],label=r'$\phi={}$'.format(factor))
        ax_w_pw.plot(w,pw,linestyle=linstyles[i%len(linstyles)],label=r'$\phi={}$'.format(factor))
    os.chdir(dir_path + directory_name)
    ax_time_h.legend()
    ax_time_h.set_title('Hamiltonian vs time')
    ax_time_h.ticklabel_format(style="sci",axis="y", scilimits=(0,0))
    ax_time_h.set_ylabel('H')
    ax_time_h.set_xlabel('t')
    fig_time_h.savefig('hamilton_v_time.png',dpi=200)
    ax_time_v_w.legend()
    ax_time_v_w.set_title('w vs time')
    ax_time_v_w.set_ylabel('w')
    ax_time_v_w.set_xlabel('t')
    fig_time_v_w.savefig('w_v_time.png',dpi=200)
    ax_time_v_u.legend()
    ax_time_v_u.set_title('u vs time')
    ax_time_v_u.set_ylabel('u')
    ax_time_v_u.set_xlabel('t')
    fig_time_v_u.savefig('u_v_time.png',dpi=200)
    ax_time_v_y1.legend()
    ax_time_v_y1.set_title(r'$y_{1}$ vs time')
    ax_time_v_y1.set_ylabel(r'$y_{1}$')
    ax_time_v_y1.set_xlabel('t')
    fig_time_v_y1.savefig('y1_v_time.png',dpi=200)
    ax_time_v_y2.legend()
    ax_time_v_y2.set_title(r'$y_{2}$ vs time')
    ax_time_v_y2.set_ylabel(r'$y_{2}$')
    ax_time_v_y2.set_xlabel('t')
    fig_time_v_y2.savefig('y2_v_time.png',dpi=200)
    ax_y1_p1.legend()
    ax_y1_p1.set_title(r'$y_{1}$ vs $p_{1}$')
    ax_y1_p1.set_ylabel(r'$p_{1}$')
    ax_y1_p1.set_xlabel(r'$y_{1}$')
    fig_y1_p1.savefig('y1_v_p1.png',dpi=200)
    ax_y2_p2.legend()
    ax_y2_p2.set_title(r'$y_{2}$ vs $p_{2}$')
    ax_y2_p2.set_ylabel(r'$p_{2}$')
    ax_y2_p2.set_xlabel(r'$y_{2}$')
    fig_y2_p2.savefig('y2_v_p2.png',dpi=200)
    ax_u_pu.legend()
    ax_u_pu.set_title(r'u vs $p_{u}$')
    ax_u_pu.set_ylabel(r'$p_{u}$')
    ax_u_pu.set_xlabel('u')
    fig_u_pu.savefig('u_v_pu.png',dpi=200)
    ax_w_pw.legend()
    ax_w_pw.set_title(r'w vs $p_{w}$')
    ax_w_pw.set_ylabel(r'$p_{w}$')
    ax_w_pw.set_xlabel('w')
    fig_w_pw.savefig('w_v_pw.png',dpi=200)
    plt.show()

def plot_path(file_name,directory_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    beta, epsilon ,factor, gamma, path, linear_combo, numpoints,qstar, radius,shot_angle, stoptime, time_series = [], [], [], [], [], [], [],[], [], [], [], []
    for f in range(len(file_name)):
        os.chdir(dir_path + directory_name)
        os.chdir(file_name[f])
        beta.append(np.load('beta.pkl',allow_pickle=True))
        epsilon.append(np.load('epsilon_matrix.pkl', allow_pickle=True))
        factor.append(np.load('factor.pkl', allow_pickle=True))
        gamma.append(np.load('gamma.pkl', allow_pickle=True))
        path.append(np.load('guessed_paths.pkl', allow_pickle=True))
        linear_combo.append(np.load('lin_combo.pkl', allow_pickle=True))
        numpoints.append(np.load('numpoints.pkl', allow_pickle=True))
        qstar.append(np.load('qstar.pkl', allow_pickle=True))
        radius.append(np.load('radius.pkl', allow_pickle=True))
        shot_angle.append(np.load('shot_angle.pkl',allow_pickle=True))
        stoptime.append(np.load('stoptime.pkl',allow_pickle=True))
        time_series.append(np.load('time_series.pkl',allow_pickle=True))
    plot_paths_energy(time_series, beta, path, epsilon, factor, dir_path, directory_name)



if __name__ == '__main__':
    directory_name = '/Data/'
    filename=['epslam04_epsin04_factor075_lam16_stoptime20_t5_no_linear_extention','epslam04_epsin04_factor09_lam16_stoptime20_t5',
              'epslam04_epsin04_factor10_lam16_stoptime20_t5_homo']
    plot_path(filename,directory_name)
    dir_path = os.path.dirname(os.path.realpath(__file__))





