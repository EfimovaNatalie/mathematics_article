import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

# definition of gate variables for HH neuron
def a_n(v):
    return (0.01 * v + 0.55) / (1 - np.exp(-0.1 * v - 5.5))
def b_n(v):
    return 0.125 * np.exp(-(v + 65) / 80)
def a_m(v):
    return (0.1 * v + 4) / (1 - np.exp(-0.1 * v - 4))
def b_m(v):
    return 4 * np.exp(-(v + 65) / 18)
def a_h(v):
    return 0.07 * np.exp(-(v + 65) / 20)
def b_h(v):
    return 1 / (1 + np.exp(-0.1 * v - 3.5))

# definition of variables for z-STDP plasticity
def w_inf(v):
    return g_slow*v
def tau_w(v):
    return tau_2 + (tau_1 - tau_2)/(1 + np.exp(-v/k_t))
def S_inf(v):
    return 1/(1 + np.exp((teta_syn - v)/k_syn))
def G(x):
    return np.sin(2*np.pi*(x - F_c))

# HH neuron parameters
V_rest = -60  # mV
V_th = 0  # mV

E_k = -77  # mV
E_Na = 50  # mV
E_L = -54.4  # mV

g_K = 36  # mSm
g_Na = 120  # mSm
g_L = 0.3  # mSm

C = 1  # microF
g_max = 10

#parameters of z-plasticity model
g_slow = 0.8
delta_I = 0#-3
I_pre = 0.7
I_post = 0.7
tau_1 = 5
tau_2 = 50
k_t = 0.5
teta_syn = 0
# current by which we excite the presynaptic neuron
I = 20*I_pre

#parameters for presynaptic feedback regulation
F_c = 0.5
alpha_pre = 0.0015

# parameters for exciting synapse:
tau_rec = 800  # ms
tau_i = 3  # ms
U = 0.5

# initial values
h_0 = 0.6
n_0 = 0.3
m_0 = 0.05
x_0 = 1
y_0 = z_0 = 0
u_0 = U
V_0 = V_rest

F_0 = 1
z_pre_0 = I_pre
z_post_0 = I_post

# STDP synapse parameters
A_plus = 0.1  # magnitude of LTP
A_minus = -A_plus  # magnitude of LTD
# print(A_minus)
tau_stdp = 20  # STDP time constant [ms]
g_0 = g_max  # g_rest #0.005*g_rest
E_rev = 0  # mV

dt = 0.01  # ms
t_start = 0  # ms
t_stop = 3000  # ms
TIME = np.linspace(t_start, t_stop, int(t_stop / dt))
t_k = t_stop-800 # ms - moment of turning on of z-plasticity mechanism
t_k = int(t_k/dt)

V_pre_list = []  # for recording of presynaptic neuron potential dynamics
V_post_list = []
pre_spikes = []  # for presynaptic spike timing recording
post_spikes = []  # for postsynaptic timing record
# lists for synaptic resources dynamic recording
X_list = []
Y_list = []
Z_list = []

V_pre_list_2 = []  # for recording of presynaptic neuron potential dynamics
V_post_list_2 = []
pre_spikes_2 = []  # for presynaptic spike timing recording
post_spikes_2 = []  # for postsynaptic timing record
# lists for synaptic resources dynamic recording
X_list_2 = []
Y_list_2 = []
Z_list_2 = []


x_trace_list = []
y_trace_list = []
G_list = []
PRE_spikes = np.zeros(len(TIME))
POST_spikes = np.zeros(len(TIME))

PRE_spikes_2 = np.zeros(len(TIME))
POST_spikes_2 = np.zeros(len(TIME))

# initial values for presynaptic neuron
V_pre = V_0
h_pre = h_0
n_pre = n_0
m_pre = m_0

# initial values for postsynaptic neuron
V_post = V_0
h_post = h_0
n_post = n_0
m_post = m_0

# setting initial values
F = F_0
z_pre = z_pre_0
z_post = z_post_0
T_pre = t_stop

# initial values for exciting synapse:
#A = 300  # variable which is proportional with synaptic current
x = x_0
y = y_0
z = z_0
u = u_0
g = g_0

x_trace = 0  # postsynaptic trace
y_trace = 0  # presynaptic trace

# initial values for presynaptic neuron
V_pre_2 = V_0
h_pre_2 = h_0
n_pre_2 = n_0
m_pre_2 = m_0

# initial values for postsynaptic neuron
V_post_2 = V_0
h_post_2 = h_0
n_post_2 = n_0
m_post_2 = m_0

# setting initial values
F_2 = F_0
z_pre_2 = z_pre_0
z_post_2 = z_post_0
T_pre_2 = t_stop

# initial values for exciting synapse:
#A = 300  # variable which is proportional with synaptic current
x_2 = x_0
y_2 = y_0
z_2 = z_0
u_2 = u_0

'''interval = 500  # ms
num_intervals = int(t_stop / interval)
frequencies_per_interval = []
intervals = []
indexes_of_times = [int(n * interval / dt) for n in range(num_intervals)]
num_spikes_per_interval = 0'''

F_list = []
z_pre_list = []

V_pre_list.append(V_pre)
V_post_list.append(V_post)
F_list.append(F)
z_pre_list.append(z_pre)
x_trace_list.append(x_trace)
y_trace_list.append(y_trace)
G_list.append(g)


F_list_2 = []
z_pre_list_2 = []

V_pre_list_2.append(V_pre_2)
V_post_list_2.append(V_post_2)
F_list_2.append(F_2)
z_pre_list_2.append(z_pre_2)

for t in TIME[1:-t_k]:
    k_pre = 0
    V_pre_old = V_pre
    V_post_old = V_post
    # stp
    x_old = x
    y_old = y
    z_old = z
    # stdp
    g_old = g
    x_old_trace = x_trace
    y_old_trace = y_trace
    # z-plasticity
    F = F_list[-1]
    z_pre_old = z_pre

    # HH-neuron
    dV_pre = (I - g_K * (n_pre ** 4) * (V_pre_old - E_k) - g_Na * (m_pre ** 3) * h_pre * (V_pre_old - E_Na) - g_L * (V_pre_old - E_L)
               - z_pre - delta_I) * dt / C
    dn_pre = (a_n(V_pre_old) * (1 - n_pre) - b_n(V_pre_old) * n_pre) * dt
    dm_pre = (a_m(V_pre_old) * (1 - m_pre) - b_m(V_pre_old) * m_pre) * dt
    dh_pre = (a_h(V_pre_old) * (1 - h_pre) - b_h(V_pre_old) * h_pre) * dt

    V_pre = V_pre_old + dV_pre
    n_pre += dn_pre
    m_pre += dm_pre
    h_pre += dh_pre

    if V_pre >= V_th and V_pre_old < V_th and dV_pre > 0:
        pre_spikes.append(t)
    # changes in synapse
    t_sp_last = max(list(filter(lambda x: x <= t, pre_spikes)), default=0)
    # print(t_sp_last)
    if abs(t_sp_last - t) <= dt:
        dx = (z_old / tau_rec) * dt - u * x_old
        dy = (-y_old / tau_i) * dt + u * x_old
    else:
        dx = (z_old / tau_rec) * dt
        dy = (-y_old / tau_i) * dt
    dz = (y_old / tau_i - z_old / tau_rec) * dt
    x = x_old + dx
    y = y_old + dy
    z = z_old + dz

    # I_syn = A*y # synaptic current due to synaptic connection between neurons
    I_syn = g_old*y_old*(E_rev - V_post_old)  # synaptic current due to synaptic connection between neurons

    # changes in postsynaptic neuron
    dV_post = (I_syn - g_K*(n_post ** 4)*(V_post_old - E_k) - g_Na*(m_post ** 3)*h_post*(V_post_old - E_Na) - g_L*(V_post_old - E_L)
                - z_post)*dt/C
    dn_post = (a_n(V_post_old) * (1 - n_post) - b_n(V_post_old) * n_post) * dt
    dm_post = (a_m(V_post_old) * (1 - m_post) - b_m(V_post_old) * m_post) * dt
    dh_post = (a_h(V_post_old) * (1 - h_post) - b_h(V_post_old) * h_post) * dt

    V_post = V_post_old + dV_post
    n_post += dn_post
    m_post += dm_post
    h_post += dh_post

    dz_pre = (alpha_pre * (I_pre - z_pre_old) + k_pre * G(F)) * dt
    z_pre = z_pre_old + dz_pre

    if V_post >= V_th and V_post_old < V_th:
        post_spikes.append(t)

    # changes in synapse
    t_pre_sp_last = max(list(filter(lambda n: n <= t, pre_spikes)), default=0)
    t_post_sp_last = max(list(filter(lambda n: n <= t, post_spikes)), default=0)
    delta = t_post_sp_last - t_pre_sp_last

    if delta < 0:
        dy_trace = -y_old_trace * dt / tau_stdp
        if abs(t_pre_sp_last - t) <= dt:
            dx_trace = -x_old_trace * dt / tau_stdp + 1
            dg = A_minus * y_old_trace
        else:
            dx_trace = -x_old_trace * dt / tau_stdp
            dg = 0
    else:
        dx_trace = -x_old_trace * dt / tau_stdp
        if abs(t_post_sp_last - t) <= dt:
            dy_trace = -y_old_trace * dt / tau_stdp + 1
            dg = A_plus * x_old_trace
        else:
            dy_trace = -y_old_trace * dt / tau_stdp
            dg = 0
    #stdp
    g = g_old + dg
    x_trace = x_old_trace + dx_trace
    y_trace = y_old_trace + dy_trace

    V_pre_list.append(V_pre)
    V_post_list.append(V_post)
    z_pre_list.append(z_pre)
    x_trace_list.append(x_trace)
    y_trace_list.append(y_trace)
    G_list.append(g)

    if V_post >= V_th and V_post_old < V_th:
        t_pre_last = max(filter(lambda x: x < t, pre_spikes), default=0)
        F = t - t_pre_last
        F_list.append(F)
    else:
        F_list.append(F)

    V_pre_old_2 = V_pre_2
    V_post_old_2 = V_post_2
    # stp
    x_old_2 = x_2
    y_old_2 = y_2
    z_old_2 = z_2
    # z-plasticity
    F_2 = F_list_2[-1]
    z_pre_old_2 = z_pre_2

    # HH-neuron
    dV_pre_2 = (I - g_K * (n_pre_2 ** 4) * (V_pre_old_2 - E_k) - g_Na * (m_pre_2 ** 3) * h_pre_2 * (V_pre_old_2 - E_Na) - g_L * (
                V_pre_old_2 - E_L)
              - z_pre_2 - delta_I) * dt / C
    dn_pre_2 = (a_n(V_pre_old_2) * (1 - n_pre_2) - b_n(V_pre_old_2) * n_pre_2) * dt
    dm_pre_2 = (a_m(V_pre_old_2) * (1 - m_pre_2) - b_m(V_pre_old_2) * m_pre_2) * dt
    dh_pre_2 = (a_h(V_pre_old_2) * (1 - h_pre_2) - b_h(V_pre_old_2) * h_pre_2) * dt

    V_pre_2 = V_pre_old_2 + dV_pre_2
    n_pre_2 += dn_pre_2
    m_pre_2 += dm_pre_2
    h_pre_2 += dh_pre_2

    if V_pre_2 >= V_th and V_pre_old_2 < V_th and dV_pre_2 > 0:
        pre_spikes_2.append(t)
    # changes in synapse
    t_sp_last_2 = max(list(filter(lambda x: x <= t, pre_spikes_2)), default=0)
    # print(t_sp_last)
    if abs(t_sp_last_2 - t) <= dt:
        dx_2 = (z_old_2 / tau_rec) * dt - u_2 * x_old_2
        dy_2 = (-y_old_2 / tau_i) * dt + u_2 * x_old_2
    else:
        dx_2 = (z_old_2 / tau_rec) * dt
        dy_2 = (-y_old_2 / tau_i) * dt
    dz_2 = (y_old_2 / tau_i - z_old_2 / tau_rec) * dt
    x_2 = x_old_2 + dx_2
    y_2 = y_old_2 + dy_2
    z_2 = z_old_2 + dz_2

    # I_syn = A*y # synaptic current due to synaptic connection between neurons
    I_syn_2 = g_max * y_old_2 * (E_rev - V_post_old_2)  # synaptic current due to synaptic connection between neurons

    # changes in postsynaptic neuron
    dV_post_2 = (I_syn_2 - g_K * (n_post_2 ** 4) * (V_post_old_2 - E_k) - g_Na * (m_post_2 ** 3) * h_post_2 * (
                V_post_old_2 - E_Na) - g_L * (V_post_old_2 - E_L)
               - z_post_2) * dt / C
    dn_post_2 = (a_n(V_post_old_2) * (1 - n_post_2) - b_n(V_post_old_2) * n_post_2) * dt
    dm_post_2 = (a_m(V_post_old_2) * (1 - m_post_2) - b_m(V_post_old_2) * m_post_2) * dt
    dh_post_2 = (a_h(V_post_old_2) * (1 - h_post_2) - b_h(V_post_old_2) * h_post_2) * dt

    V_post_2 = V_post_old_2 + dV_post_2
    n_post_2 += dn_post_2
    m_post_2 += dm_post_2
    h_post_2 += dh_post_2

    dz_pre_2 = (alpha_pre * (I_pre - z_pre_old_2) + k_pre * G(F_2)) * dt
    z_pre_2 = z_pre_old_2 + dz_pre_2

    if V_post_2 >= V_th and V_post_old_2 < V_th:
        post_spikes_2.append(t)

    # changes in synapse
    t_pre_sp_last_2 = max(list(filter(lambda n: n <= t, pre_spikes_2)), default=0)
    t_post_sp_last_2 = max(list(filter(lambda n: n <= t, post_spikes_2)), default=0)

    V_pre_list_2.append(V_pre_2)
    V_post_list_2.append(V_post_2)
    z_pre_list_2.append(z_pre_2)

    if V_post_2 >= V_th and V_post_old_2 < V_th:
        t_pre_last_2 = max(filter(lambda x: x < t, pre_spikes_2), default=0)
        F_2 = t - t_pre_last_2
        F_list_2.append(F_2)
    else:
        F_list_2.append(F_2)

if len(pre_spikes) > 2:
    periods = [pre_spikes[i + 1] - pre_spikes[i] for i in range(len(pre_spikes) - 1)]
    T_pre = sum(periods) / len(periods)
F_list_new = [x / T_pre for x in F_list]

if len(pre_spikes_2) > 2:
    periods_2 = [pre_spikes_2[i+1] - pre_spikes_2[i] for i in range(len(pre_spikes_2) - 1)]
    T_pre_2 = sum(periods_2)/len(periods_2)
F_list_new_2 = [x/T_pre_2 for x in F_list_2]

for t in TIME[-t_k::]:
    V_pre_old = V_pre
    V_post_old = V_post
    # stp
    x_old = x
    y_old = y
    z_old = z
    # stdp
    g_old = g
    x_old_trace = x_trace
    y_old_trace = y_trace
    # z-plasticity
    F = F_list_new[-1]
    k_pre = 0.5
    z_pre_old = z_pre

    # HH-neuron
    dV_pre = (I - g_K * (n_pre ** 4) * (V_pre_old - E_k) - g_Na * (m_pre ** 3) * h_pre * (
                V_pre_old - E_Na) - g_L * (V_pre_old - E_L) - z_pre - delta_I) * dt / C
    dn_pre = (a_n(V_pre_old) * (1 - n_pre) - b_n(V_pre_old) * n_pre) * dt
    dm_pre = (a_m(V_pre_old) * (1 - m_pre) - b_m(V_pre_old) * m_pre) * dt
    dh_pre = (a_h(V_pre_old) * (1 - h_pre) - b_h(V_pre_old) * h_pre) * dt

    V_pre = V_pre_old + dV_pre
    n_pre += dn_pre
    m_pre += dm_pre
    h_pre += dh_pre

    if V_pre >= V_th and V_pre_old < V_th and dV_pre > 0:
        pre_spikes.append(t)
        # PRE_spikes[i] = 1

    # changes in synapse
    t_sp_last = max(list(filter(lambda x: x <= t, pre_spikes)), default=0)
    # print(t_sp_last)
    if abs(t_sp_last - t) <= dt:
        dx = (z_old / tau_rec) * dt - u * x_old
        dy = (-y_old / tau_i) * dt + u * x_old
    else:
        dx = (z_old / tau_rec) * dt
        dy = (-y_old / tau_i) * dt
    dz = (y_old / tau_i - z_old / tau_rec) * dt
    x = x_old + dx
    y = y_old + dy
    z = z_old + dz

    I_syn = g_old * y_old * (E_rev - V_post_old)   # synaptic current due to synaptic connection between neurons

    # changes in postsynaptic neuron
    dV_post = (I_syn - g_K*(n_post**4)*(V_post_old - E_k) - g_Na*(m_post**3)*h_post*(V_post_old - E_Na) - g_L*(V_post_old - E_L)
               - z_post) * dt / C
    dn_post = (a_n(V_post_old) * (1 - n_post) - b_n(V_post_old) * n_post) * dt
    dm_post = (a_m(V_post_old) * (1 - m_post) - b_m(V_post_old) * m_post) * dt
    dh_post = (a_h(V_post_old) * (1 - h_post) - b_h(V_post_old) * h_post) * dt

    V_post = V_post_old + dV_post
    n_post += dn_post
    m_post += dm_post
    h_post += dh_post

    # z-plasticity
    dz_pre = (alpha_pre * (I_pre - z_pre_old) + k_pre * G(F)) * dt
    z_pre = z_pre_old + dz_pre

    if V_post >= V_th and V_post_old < V_th:
        post_spikes.append(t)
        # POST_spikes[i] = 1
        #num_spikes_per_interval += 1

    '''if i in indexes_of_times:
        print(t, num_spikes_per_interval)
        frequency_per_interval = 1000 * num_spikes_per_interval / interval  # Hz
        frequencies_per_interval.append(frequency_per_interval)
        intervals.append(t)
        num_spikes_per_interval = 0'''

    # changes in synapse
    t_pre_sp_last = max(list(filter(lambda n: n <= t, pre_spikes)), default=0)
    t_post_sp_last = max(list(filter(lambda n: n <= t, post_spikes)), default=0)
    delta = t_post_sp_last - t_pre_sp_last

    if delta < 0:
        dy_trace = -y_old_trace * dt / tau_stdp
        if abs(t_pre_sp_last - t) <= dt:
            dx_trace = -x_old_trace * dt / tau_stdp + 1
            dg = A_minus * y_old_trace
        else:
            dx_trace = -x_old_trace * dt / tau_stdp
            dg = 0
    else:
        dx_trace = -x_old_trace * dt / tau_stdp
        if abs(t_post_sp_last - t) <= dt:
            dy_trace = -y_old_trace * dt / tau_stdp + 1
            dg = A_plus * x_old_trace
        else:
            dy_trace = -y_old_trace * dt / tau_stdp
            dg = 0
    # stdp
    g = g_old + dg
    x_trace = x_old_trace + dx_trace
    y_trace = y_old_trace + dy_trace

    V_pre_list.append(V_pre)
    V_post_list.append(V_post)
    z_pre_list.append(z_pre)
    x_trace_list.append(x_trace)
    y_trace_list.append(y_trace)
    G_list.append(g)

    if V_post >= V_th and V_post_old < V_th:
        t_pre_last = max(list(filter(lambda x: x < t, pre_spikes)), default=0)
        if len(pre_spikes) > 2:
            T_pre = t_pre_last - pre_spikes[-2]
        else:
            print(len(pre_spikes))
        #    T_pre = t - t_pre_last
        # T_pre = t_pre_last - t_pre_spikes[-2]
        F = (t - t_pre_last) / T_pre
        F_list_new.append(F)
    else:
        F_last = F_list_new[-1]
        F_list_new.append(F_last)

    V_pre_old_2 = V_pre_2
    V_post_old_2 = V_post_2
    # stp
    x_old_2 = x_2
    y_old_2 = y_2
    z_old_2 = z_2
    # z-plasticity
    F_2 = F_list_new_2[-1]
    z_pre_old_2 = z_pre_2

    # HH-neuron
    dV_pre_2 = (I - g_K * (n_pre_2 ** 4) * (V_pre_old_2 - E_k) - g_Na * (m_pre_2 ** 3) * h_pre_2 * (
                V_pre_old_2 - E_Na) - g_L * (
                        V_pre_old_2 - E_L)
                - z_pre_2 - delta_I) * dt / C
    dn_pre_2 = (a_n(V_pre_old_2) * (1 - n_pre_2) - b_n(V_pre_old_2) * n_pre_2) * dt
    dm_pre_2 = (a_m(V_pre_old_2) * (1 - m_pre_2) - b_m(V_pre_old_2) * m_pre_2) * dt
    dh_pre_2 = (a_h(V_pre_old_2) * (1 - h_pre_2) - b_h(V_pre_old_2) * h_pre_2) * dt

    V_pre_2 = V_pre_old_2 + dV_pre_2
    n_pre_2 += dn_pre_2
    m_pre_2 += dm_pre_2
    h_pre_2 += dh_pre_2

    if V_pre_2 >= V_th and V_pre_old_2 < V_th and dV_pre_2 > 0:
        pre_spikes_2.append(t)
    # changes in synapse
    t_sp_last_2 = max(list(filter(lambda x: x <= t, pre_spikes_2)), default=0)
    # print(t_sp_last)
    if abs(t_sp_last_2 - t) <= dt:
        dx_2 = (z_old_2 / tau_rec) * dt - u_2 * x_old_2
        dy_2 = (-y_old_2 / tau_i) * dt + u_2 * x_old_2
    else:
        dx_2 = (z_old_2 / tau_rec) * dt
        dy_2 = (-y_old_2 / tau_i) * dt
    dz_2 = (y_old_2 / tau_i - z_old_2 / tau_rec) * dt
    x_2 = x_old_2 + dx_2
    y_2 = y_old_2 + dy_2
    z_2 = z_old_2 + dz_2

    # I_syn = A*y # synaptic current due to synaptic connection between neurons
    I_syn_2 = g_max * y_old_2 * (E_rev - V_post_old_2)  # synaptic current due to synaptic connection between neurons

    # changes in postsynaptic neuron
    dV_post_2 = (I_syn_2 - g_K * (n_post_2 ** 4) * (V_post_old_2 - E_k) - g_Na * (m_post_2 ** 3) * h_post_2 * (
            V_post_old_2 - E_Na) - g_L * (V_post_old_2 - E_L)
                 - z_post_2) * dt / C
    dn_post_2 = (a_n(V_post_old_2) * (1 - n_post_2) - b_n(V_post_old_2) * n_post_2) * dt
    dm_post_2 = (a_m(V_post_old_2) * (1 - m_post_2) - b_m(V_post_old_2) * m_post_2) * dt
    dh_post_2 = (a_h(V_post_old_2) * (1 - h_post_2) - b_h(V_post_old_2) * h_post_2) * dt

    V_post_2 = V_post_old_2 + dV_post_2
    n_post_2 += dn_post_2
    m_post_2 += dm_post_2
    h_post_2 += dh_post_2

    dz_pre_2 = (alpha_pre * (I_pre - z_pre_old_2) + k_pre * G(F_2)) * dt
    z_pre_2 = z_pre_old_2 + dz_pre_2

    if V_post_2 >= V_th and V_post_old_2 < V_th:
        post_spikes_2.append(t)

    # changes in synapse
    t_pre_sp_last_2 = max(list(filter(lambda n: n <= t, pre_spikes_2)), default=0)
    t_post_sp_last_2 = max(list(filter(lambda n: n <= t, post_spikes_2)), default=0)

    V_pre_list_2.append(V_pre_2)
    V_post_list_2.append(V_post_2)
    z_pre_list_2.append(z_pre_2)

    if V_post_2 >= V_th and V_post_old_2 < V_th:
        t_pre_last_2 = max(list(filter(lambda x: x < t, pre_spikes_2)), default=0)
        if len(pre_spikes_2) > 2:
            T_pre_2 = t_pre_last_2 - pre_spikes_2[-2]
        else:
            print(len(pre_spikes_2))
        #    T_pre = t - t_pre_last
        # T_pre = t_pre_last - t_pre_spikes[-2]
        F_2 = (t - t_pre_last_2) / T_pre_2
        F_list_new_2.append(F_2)
    else:
        F_last_2 = F_list_new_2[-1]
        F_list_new_2.append(F_last_2)

F_c_list = np.ones(len(TIME)) * (F_c)

plt.figure(figsize=(10, 10))
mpl.rcParams['font.size'] = 20
plt.plot(TIME, V_pre_list, label='V_1(t) for presynaptic HH neuron', c=(0.01, 0.01, 0.8), lw=2)
plt.plot(TIME, V_pre_list_2, label='V_2(t) for presynaptic HH neuron', c=(1, 0.01, 0.01), lw=2)
plt.plot(TIME, V_post_list, label='V(t) for postsynaptic HH neuron', c=(1, 0.55, 0), lw=2)
plt.ylabel("membrane potential, [mV]")
plt.xlabel("time, [ms]")
plt.legend()
plt.figure(figsize=(10, 10))
mpl.rcParams['font.size'] = 20
#plt.plot(TIME, F_list_new, label="Ф(t)", c=(0.01, 0.01, 0.8), lw=2)
#plt.plot(TIME, F_c_list, label="Ф_c", c=(1, 0.55, 0), lw=2)
plt.plot(TIME, z_pre_list, label=r"$z_{pre}(t)$", c=(0.01, 0.01, 0.8), lw=2)
plt.xlabel("time, [ms]")
#plt.ylabel("Ф(t)")
plt.title("$z_{pre}(t)$")
#plt.title("Relative spiking phase")
plt.legend()

plt.figure(figsize=(10, 10))
mpl.rcParams['font.size'] = 20
plt.plot(TIME, F_list_new, label=r"$Ф(t)$", c=(0.01, 0.01, 0.8), lw=2)
plt.plot(TIME, F_list_new_2, label=r"$Ф(t)$ without STDP", c=(1, 0.01, 0.01), lw=2)
plt.plot(TIME, F_c_list, label=r"$Ф_{c}$", c=(1, 0.55, 0), lw=2)
plt.xlabel("time, [ms]")
plt.ylabel(r"relative spiking phase, $Ф(t)$")
#plt.title("Relative spiking phase")
plt.legend()

'''plt.figure(figsize=(10, 10))
plt.plot(TIME, x_trace_list, label='presynaptic trace', c=(0.01, 0.01, 0.8), lw=2)
plt.plot(TIME, y_trace_list, label='postsynaptic trace', c=(1, 0.55, 0), lw=2)
#plt.plot(TIME, G_list, label='synaptic conductance')
#plt.plot(TIME, PRE_spikes, label='pre_spikes')
#plt.plot(TIME, POST_spikes, label='post_spikes')
plt.legend()
plt.figure(figsize=(10, 14))
plt.scatter(intervals, frequencies_per_interval, label="frequencies per interval", s=20, c="red")
plt.xlabel("time, [ms]")
plt.ylabel("firing rate, [Hz]")
mpl.rcParams['font.size'] = 20
plt.legend()'''

fourier = np.fft.fft(V_post_list)/len(V_post_list)
fourier_1 = np.fft.fft(V_pre_list)/len(V_pre_list)
freq = np.fft.fftfreq(np.array(V_post_list).size, d=dt)
plt.figure(figsize=(10, 10))
mpl.rcParams['font.size'] = 20
plt.plot(1000*freq, np.abs(fourier), label=r"for $V_{post}$", c=(1, 0.55, 0), lw=2)#, freq, sp.imag)
plt.plot(1000*freq, np.abs(fourier_1), label=r"for $V_{pre}$", c=(0.01, 0.01, 0.8), lw=2, linestyle='dotted')
plt.xlabel("frequency, [Hz]")
plt.ylabel("fourier transform")
#plt.plot(freq_1, fourier_1)
plt.xlim(0, 500)
#plt.ylim(-0.5, 20)
plt.legend()

plt.show()