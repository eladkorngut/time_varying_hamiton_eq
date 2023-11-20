import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.optimize import root
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.integrate import simps

# hamiltonian = lambda R0,w,pw,phi: 2.0*w*(-1.0 + np.exp(-pw/2.0) + (-1.0 + np.exp(pw/2.0))*R0*(-1 + 2.0*w)*(-1.0 + phi))
hamiltonian = lambda R0,x,p: -(((-1 + np.exp(p))*(1 + np.exp(p)*R0*(-1 + x))*x)/np.exp(p))

energy_numeric = lambda R0,x,p,t0,tf,xi,time_series: np.where(np.logical_or(time_series<t0,time_series> tf) ,hamiltonian(R0,x,p),hamiltonian(R0*xi,x,p))

pw_pertubation = lambda R0,w,phi,sigma: 2*(-np.log(4*R0*w) + np.log((sigma +2*w*(1 +R0*(-1 + 2*w)*(-1 + phi)) -
            np.sqrt((sigma +2*w*(1 +R0*(-1 + 2*w)*(-1 + phi)))**2- 16*R0*w**2*(-1 + 2*w)*(-1 + phi)))/((-1 + 2*w)*
           (-1 + phi))))
# pw_unpertubated = lambda w,beta: -2*np.log(beta*(1-2*w))
pw_unpertubated = lambda w,beta: -np.log(beta*(1-w))

pw_energy_math = lambda w,R0,sigma: 2*(-np.log(4*R0) + np.log((-2*(1 + R0)*w + 4*R0*w**2 - sigma +
                       np.sqrt(16*R0*w**2*(-1 + 2*w) + (2*w*(1 + R0 - 2*R0*w) + sigma)**2))/(w*(-1 + 2*w))))

dwdt = lambda q,R0:-((q[0]*(1 + np.exp(q[1])*R0*(-1 + 2*q[0])))/np.exp(q[1]/2))

dtdw = lambda q,R0: 1/( -((q[0]*(1 + np.exp(q[1])*R0*(-1 + 2*q[0])))/np.exp(q[1]/2)) )


theory_energy = lambda R0,T,epsilon: (np.exp((-1 - R0)*T)*epsilon*(-(np.exp(T + R0*T)*(-1 + R0)**2*R0) + np.sqrt(
    np.exp(2*(1 + R0)*T)*(-1 + R0)**4*R0**2*np.cosh(((-1 + R0)*T)/2.)**2))*Csch((T - R0*T)/2)**2)/(2*R0**2)

def energy_time_series(R0,epsilon,time_series,t0,tf):
    return np.where(np.logical_or(time_series<t0,time_series> tf) ,0.0,theory_energy(R0,tf-t0,epsilon))


# energy_theory_time lambda R0,T,epsilon,t0,tau,time_series: 0.0 if time_series<
# path_theory = lambda x,R0,epsilon,T:np.log(-(x + R0*x*(1 - epsilon) - R0*x**2*(1 - epsilon) +
#                 (R0*epsilon*np.tanh((T - R0*T)/2)**2)/4 - np.sqrt(4*R0*(-1 + x)*x**2*(1 - epsilon) +
#                 (x + R0*x*(1 - x*(1 - epsilon) - epsilon) + (R0*epsilon*np.tanh((T - R0*T)/2)**2)/4)**2))
#                 /(2*R0*(-1 + x)*x*(1 - epsilon)))
path_theory_cat = lambda x,R0,epsilon,T: epsilon - np.log(R0 - R0*x) + ((-1 + R0)**2*epsilon*(1/np.sinh((T - R0*T)/2))**2
                *np.sinh(((-1 + R0)*T)/4)**2)/(R0*(1 + R0*(-1 + x))*x)



def combined_paths(w,R0,sigma):
    return pw_unpertubated(w,R0)+pw_energy_math(w,R0,sigma)

# target_function = lambda pw,w,e,R0,phi: 2.0*w*(-1.0 + np.exp(-pw/2.0) + (-1.0 + np.exp(pw/2.0))*R0*(-1 + 2.0*w)*(-1.0 + phi)) + e
target_function = lambda pw,w,e,R0,phi: 2*w*( -1+1/pw+(-1+pw)*R0*(-1+2*w)*(-1+phi) ) + e

# action_theory = lambda R0,T,epsilon: -1 + 1/R0 - (T*epsilon*(-(np.exp(T + R0*T)*(-1 + R0)**2*R0) + np.sqrt(np.exp(2*(1 +
#          R0)*T)*(-1 + R0)**4*R0**2*np.cosh(((-1 + R0)*T)/2)**2))*(1/np.sinh((T - R0*T)/2))**2)/(2*np.exp((1 + R0)*T)*R0**2)\
#          + np.log(R0) - ((-1 + R0)*epsilon*((1/np.sinh((T - R0*T)/2))**2*(-2*np.arctanh(np.sqrt(1 - 4*(1/np.sinh((T - R0*T)/2))**2*
#         np.sinh(((-1 + R0)*T)/4)**2)) - np.log(-1 - np.sqrt(1 - 4*(1/np.sinh((T - R0*T)/2))**2*np.sinh(((-1 + R0)*T)/4.)**2)) +
#         np.log(-1 + np.sqrt(1 - 4*(1/np.sinh((T - R0*T)/2))**2*np.sinh(((-1 + R0)*T)/4)**2)))*np.sinh(((-1 + R0)*T)/4)**2 +
#         np.sqrt(1 - 4*(1/np.sinh((T - R0*T)/2))**2*np.sinh(((-1 + R0)*T)/4)**2)))/R0

Csch = lambda x: 1/np.sinh(x)

action_theory = lambda R0,T,epsilon: -1 + 1/R0 - (T*epsilon*(-(np.exp(T + R0*T)*(-1 + R0)**2*R0) +
                np.sqrt(np.exp(2*(1 + R0)*T)*(-1 + R0)**4*R0**2*np.cosh(((-1 + R0)*T)/2)**2))*Csch((T - R0*T)/2)**2)/(
                2*np.exp((1 + R0)*T)*R0**2) + np.log(R0) - ((-1 + R0)*epsilon*(Csch((T - R0*T)/2)**2*(-2*np.arctanh
                (np.sqrt(1 -4*Csch((T - R0*T)/2)**2*np.sinh(((-1 + R0)*T)/4)**2)) +np.log((-1 +np.sqrt(1 -4*Csch((T - R0*T)/2)
                **2*np.sinh(((-1 + R0)*T)/4)**2))/(-1 - np.sqrt(1 -4*Csch((T - R0*T)/2)**2*np.sinh(((-1 + R0)*T)/4)**2))))
                *np.sinh(((-1 + R0)*T)/4)**2 +np.sqrt(1 - 4*Csch((T - R0*T)/2.)**2*np.sinh(((-1 + R0)*T)/4)**2)))/R0


def theortical_catastrophe_path(R0,xi,tc):
    dxdp = lambda x,E0,xi,R0:-(1/np.sqrt(4*R0*(-1 + x)*x**2*xi + (E0 + x + R0*x*(xi - x*xi))**2))
    xupper = lambda E0,xi,R0: (1 + (-1 + np.sqrt((-1 + 4*E0*R0 + (-2 + R0)*R0*(-1 + xi) + xi)/(-1 + xi)))/R0)/2
    xlower = lambda E0,xi,R0: (1 - (1 + np.sqrt((-1 + 4*E0*R0 + (-2 + R0)*R0*(-1 + xi) + xi)/(-1 + xi)))/R0)/2
    cat_time_theory = lambda E0,xi,R0: quad(dxdp,xlower(E0,xi,R0),xupper(E0,xi,R0),args=(E0,xi,R0))
    # target_function = lambda E0,xi,R0,tc: cat_time_theory(E0,xi,R0)-tc
    target_function = lambda E0,xi,R0: cat_time_theory(E0,xi,R0)
    sol = root(target_function,0.0001,args=(xi,R0))
    return sol


def find_intersection(w,R0,sigma):
    return fsolve(combined_paths,0.0,args={R0,})

def find_solution(R0,phi,duration):
    num_points = 100
    energy = np.linspace(-1.0,0.001,num_points)
    x0,pstar = (1/2)*(1-1/R0),-2*np.log(R0)
    w_theory = np.linspace(x0,0.001,num_points)
    # pw_theory_reg = -2*np.log(R0*(1-2*w_theory))
    pw_theory_reg = 1/(R0-2*R0*w_theory)
    pw_theory_org = -2*np.log(R0*(1-2*w_theory))
    array_best_path,array_intersection_points = [],[]
    array_of_diff_p, array_intersection_points_w = [], []
    paths = []
    for e in energy:
        count = 0
        p_array = []
        for w in w_theory:
            # p_array = root(target_function, [e, w], args=(R0, phi))
            p_sol = root(target_function, pw_theory_reg[count], args=(w,e,R0, phi) )
            p_array.append(float(2*np.log(p_sol.x)))
            count = count + 1
        paths.append(p_array)
        sort_array = np.argsort(np.abs(p_array-pw_theory_org))
        array_intersection_points_w.append((p_array[sort_array[0]],p_array[sort_array[1]]))
        array_of_diff_p.append( np.abs(p_array[sort_array[0]]-pw_theory_org[sort_array[0]]) +
                                np.abs(p_array[sort_array[1]]-pw_theory_org[sort_array[1]]) )
    # best_path = array_of_diff_p.index(min(array_of_diff_p))
    # array_best_path.append(paths[best_path])
    # array_intersection_points.append(array_intersection_points_w[best_path])
    degree = 20
    predicted_time = []
    for intersection,pw_path,e in zip(array_intersection_points_w,paths,energy):
        poly = PolynomialFeatures(degree=degree)
        w_reshaped = w_theory.reshape(-1, 1)
        w_poly = poly.fit_transform(w_reshaped)
        poly_model = LinearRegression()
        poly_model.fit(w_poly, pw_path)

        def predicted_pw_function(x):
            x_poly = poly.transform([[x]])
            return poly_model.predict(x_poly)

        dtdw = lambda w, R0: 1 / (-((w * (1 + np.exp(predicted_pw_function(w)) * R0 * (-1 + 2 * w)))
                                    / np.exp(predicted_pw_function(w) / 2)))

        predicted_time.append(quad(dtdw,min(intersection),max(intersection),args=(R0)))
    diff = np.abs(np.array(predicted_time)[:,0] - duration)
    path_index=np.nanargmin(diff)
    return w_theory,paths[path_index],energy[path_index]


def plot_paths_energy(t,lam,p,xi,tau,dir_path,directory_name,t0,time):
    # phi = 1-xi[0]
    phi = 1-xi
    # fig_time_v_w, ax_time_v_w = plt.subplots()
    # fig_u_pu, ax_u_pu = plt.subplots()
    fig_w_pw, ax_w_pw = plt.subplots()
    # fig_w_t, ax_w_t = plt.subplots()
    # fig_pw_t, ax_pw_t = plt.subplots()
    # fig_energy,ax_energy = plt.subplots()
    linstyles=['-','--',':','-.']
    # for i,(time_series,beta,path,factor,t0,x,tcat) in enumerate(zip(t,lam,p,f,tau,xi,tf)):
    #     # w,u,pw,pu = (path[:, 0]+path[:, 1])/2,(path[:, 0]-path[:, 1])/2,(path[:, 2] + path[:, 3]) / 2,(path[:, 2] - path[:, 3]) / 2
    #     # w,u,pw,pu = (path[:, 0]+path[:, 1])/2,(path[:, 0]-path[:, 1])/2,(path[:, 2] + path[:, 3]) / 2,(path[:, 2] - path[:, 3]) / 2
    #     # timed_factor = np.ones(np.size(time_series))
    #     # timed_factor[np.logical_and(time_series > 0.1, time_series < 1.1)] = factor
    #     # timed_beta = beta * timed_factor
    #     # ax_u_pu.plot(u,pu,linestyle=linstyles[i%len(linstyles)],label='')
    #     # ax_w_pw.plot(path[:, 0],path[:, 1],linestyle=linstyles[i%len(linstyles)],label='Sim')
    #     # ax_w_pw.plot(path[:, 0],-2*np.log(beta*(1-2*path[:, 0])),linestyle=linstyles[i%len(linstyles)],label='Homo')
    #     ax_w_pw.plot(path[:, 0],path[:, 1],linestyle='-',label='Sim',linewidth='3')
    #     ax_w_t.plot(time_series,path[:, 0],linestyle='-',label='Sim',linewidth='3')
    #     ax_pw_t.plot(time_series,path[:, 1],linestyle='-',label='Sim',linewidth='3')
    #     # ax_w_pw.plot(path[:, 0],-2*np.log(beta*(1-2*path[:, 0])),linestyle='--',label='Homo',linewidth='3')
    #     # R0 = np.where(np.logical_and(time_series > t0, time_series < tcat),x * beta,beta )
    #     phi = np.where(np.logical_and(time_series > t0, time_series < tcat),1.0-x,0.0)
    #     ax_energy.plot(time_series,hamiltonian(beta,path[:, 0],path[:, 1],phi),linestyle='-',label='H',linewidth='3')
    os.chdir(dir_path + directory_name)
    # ax_u_pu.legend()
    # ax_u_pu.set_title(r'u vs $p_{u}$')
    # ax_u_pu.set_ylabel(r'$p_{u}$')
    # ax_u_pu.set_xlabel('u')
    # fig_u_pu.savefig('u_v_pu.png',dpi=200)
    # ax_w_pw.plot(p[0][0:6909, 0], p[0][0:6909, 1], linestyle='-', label='Sim', linewidth='3')
    ax_w_pw.plot(p[:, 0], p[:, 1], linestyle='-', label='Sim', linewidth='3')
    # ax_w_pw.plot(p[0:6909, 0],-2*(lam-1)+4*p[0:6909, 0], linestyle=':', label='Aprox', linewidth='3')
    # ax_w_pw.plot(p[0][0:6909, 0],pw_unpertubated(p[0][0:6909, 0],lam), linestyle='--', label='Theory unperturbed', linewidth='3')
    ax_w_pw.plot(p[:, 0],pw_unpertubated(p[:, 0],lam), linestyle='--', label='Theory unperturbed', linewidth='3')
    # ax_w_pw.plot(p[0:5500, 0],path_theory(p[0:5500, 0],lam,phi,t0),linestyle=':', label='Theory perturbed', linewidth='3')
    # ax_w_pw.plot(p[(time>t0) & (time<t0+tau),0],path_theory_cat(p[(time>t0) & (time<t0+tau),0],lam,tau,phi),linestyle=':', label='Theory cat', linewidth='3')
    path_theory = path_theory_cat(p[:, 0],lam,phi,tau)
    ax_w_pw.plot(p[3000:5500, 0],path_theory[3000:5500],linestyle=':', label='Theory perturbed', linewidth='3')
    # w_theory_exact,pw_theory_exact,energy = find_solution(lam[0],phi,t0[0])
    # ax_w_pw.plot(w_theory_exact,pw_theory_exact, linestyle=':', label='Theory', linewidth='3')
    ax_w_pw.legend()
    ax_w_pw.set_title(r'$p$ vs x for $R_{0}$='+str(lam)+', T='+str(tau))
    # ax_w_pw.set_title(r'$p_{w}$ vs w for $R_{0}$='+str(lam[0])+', T='+str(t0[0]))
    ax_w_pw.set_ylabel(r'$p$')
    ax_w_pw.set_xlabel('x')
    fig_w_pw.savefig('w_v_pw.png',dpi=200)
    # ax_w_t.legend()
    # ax_w_t.set_title(r'w vs time')
    # ax_w_t.set_ylabel(r'$w$')
    # ax_w_t.set_xlabel('time')
    # fig_w_t.savefig('w_v_t.png',dpi=200)
    # ax_pw_t.legend()
    # ax_pw_t.set_title(r'$p_{w}$ vs time')
    # ax_w_t.set_ylabel(r'$p_{w}$')
    # ax_w_t.set_xlabel('time')
    # fig_w_t.savefig('pw_v_t.png',dpi=200)
    # ax_energy.legend()
    # ax_energy.set_title('Energy vs time')
    # ax_energy.set_ylabel('H')
    # ax_energy.set_xlabel('time')
    # fig_energy.savefig('H_v_t.png',dpi=200)
    plt.show()


def plot_action(time_series, beta, path, factor, tau, dir_path, directory_name,xi,tf):
    epsilon = np.ones(len(xi))-xi
    cat_duration = np.array(tf) - np.array(tau)
    beta = np.array(beta)
    fig_s_v_eps, ax_s_v_eps = plt.subplots()
    fig_energy_v_time, ax_energy_v_time = plt.subplots()
    os.chdir(dir_path + directory_name)
    action,energy_vec = [],[]
    for R0,p,t,t0,tfinal,fac in zip(beta,path,time_series,tau,tf,factor):
        action.append(simps((p[:,1]),p[:,0])-np.max(energy_numeric(R0,p[:,0],p[:,1],t0,tfinal,fac,t))*(tfinal-t0))
        # energy_vec.append( hamiltonian(beta[0],p[:,0],p[:,1]) )
        ax_energy_v_time.plot(t,energy_numeric(R0,p[:,0],p[:,1],t0,tfinal,fac,t),linestyle='-', label='Sim', linewidth='2')
        ax_energy_v_time.plot(t,energy_time_series(R0,1-fac,t,t0,tfinal),linestyle='--', label='Theory', linewidth='2')
    epsilon_theory = np.linspace(min(epsilon),max(epsilon),100)
    act = action_theory(beta[0],cat_duration[0],epsilon_theory)
    ax_s_v_eps.plot(epsilon_theory,act, linestyle='-', label='Theory', linewidth='3')
    ax_s_v_eps.plot(epsilon,action,linestyle='none', label='Sim', linewidth='3',Marker='o')
    ax_s_v_eps.set_ylabel('S')
    ax_s_v_eps.set_xlabel(r'$\epsilon$')
    ax_s_v_eps.set_title(r'S vs $\epsilon$ for R0={}, T={}'.format(beta[0],tf[0]-tau[0]))
    ax_s_v_eps.legend(['Sim','Theory'])
    ax_energy_v_time.set_ylabel('H')
    ax_energy_v_time.set_xlabel('Time')
    ax_energy_v_time.set_title(r'Energy vs Time for R0={}, T={}'.format(beta[0],tf[0]-tau[0]))
    ax_energy_v_time.legend(['Sim','Theory'])
    fig_s_v_eps.savefig('action_v_eps_cat{}_r{}.png'.format(tf[0]-tau[0],beta[0]),dpi=200)
    fig_energy_v_time.savefig('H_v_t_cat{}_r{}.png'.format(tf[0]-tau[0],beta[0]),dpi=200)
    plt.show()
    print('this no love song')


def plot_path(file_name,directory_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    beta ,factor, path, numpoints,qstar, radius,stoptime, time_series,tau,tf,xi = [], [], [], [], [], [], [],[], [],[],[]
    for f in range(len(file_name)):
        os.chdir(dir_path + directory_name)
        os.chdir(file_name[f])
        beta.append(np.load('beta.pkl',allow_pickle=True))
        path.append(np.load('guessed_paths.pkl', allow_pickle=True))
        numpoints.append(np.load('numpoints.pkl', allow_pickle=True))
        qstar.append(np.load('qstar.pkl', allow_pickle=True))
        radius.append(np.load('radius.pkl', allow_pickle=True))
        stoptime.append(np.load('stoptime.pkl',allow_pickle=True))
        time_series.append(np.load('time_series.pkl',allow_pickle=True))
        tau.append(np.load('t0.pkl',allow_pickle=True))
        factor.append(np.load('xi.pkl',allow_pickle=True))
        tf.append(np.load('tf.pkl',allow_pickle=True))
        xi.append(np.load('xi.pkl',allow_pickle=True))
    # plot_paths_energy(time_series, beta, path, factor, tau, dir_path, directory_name,xi,tf)
    plot_action(time_series, beta, path, factor, tau, dir_path, directory_name,xi,tf)


if __name__ == '__main__':
    directory_name = '/Data1d/'
    # filename=['epslam04_epsin04_factor075_lam16_stoptime20_t5_no_linear_extention','epslam04_epsin04_factor09_lam16_stoptime20_t5',
    #           'epslam04_epsin04_factor10_lam16_stoptime20_t5_homo']
    # filename=['path_R0_1.3_stoptime_100.0_t0_50.0_tf_50.5_xi_0.5_rad_2.7879e-08']
    filename=['path_R0_1.3_stoptime_40.0_t0_17.0_tf_19.0_xi_0.95_rad_0.001078','path_R0_1.3_stoptime_40.0_t0_17.0_tf_19.0_xi_0.925_rad_0.0010273',
              'path_R0_1.3_stoptime_40.0_t0_17.0_tf_19.0_xi_0.9_rad_0.00098','path_R0_1.3_stoptime_40.0_t0_17.0_tf_19.0_xi_0.875_rad_0.0009345',
              'path_R0_1.3_stoptime_40.0_t0_17.0_tf_19.0_xi_0.85_rad_0.0008915']

    # filename=['path_R0_1.3_stoptime_40.0_t0_17.0_tf_19.0_xi_0.9_rad_0.00098','path_R0_1.3_stoptime_40.0_t0_17.0_tf_19.0_xi_0.85_rad_0.0008915',
    #           'path_R0_1.3_stoptime_40.0_t0_17.0_tf_19.0_xi_0.95_rad_0.001078','path_R0_1.3_stoptime_40.0_t0_17.0_tf_19.0_xi_0.875_rad_0.0009345',
    #           'path_R0_1.3_stoptime_40.0_t0_17.0_tf_19.0_xi_0.925_rad_0.0010273']


    # find_solution(1.1,0.5,1.0)
    plot_path(filename,directory_name)
    dir_path = os.path.dirname(os.path.realpath(__file__))