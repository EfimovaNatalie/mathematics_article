import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# definition of gate variables for Hodgkin Huxley neuron modeling
def a_n(v):
    return (0.01*v + 0.55)/(1 - np.exp(-0.1*v - 5.5))
def b_n(v):
    return 0.125*np.exp(-(v + 65)/80)
def a_m(v):
    return (0.1*v + 4)/(1 - np.exp(-0.1*v - 4))
def b_m(v):
    return 4 * np.exp(-(v + 65)/ 18)
def a_h(v):
    return 0.07*np.exp(-(v + 65)/20)
def b_h(v):
    return 1/(1 + np.exp(-0.1*v - 3.5))

font_mode = 'work'
lbl_fontsize_dict = {'work' : plt.rcParams['font.size'], 'present' : 20}
ticks_fontsize_dict = {'work' : plt.rcParams['font.size'], 'present' : 20}

def get_fig(xlbl, ylbl, title=None, xscl='linear', yscl='linear',
			xlbl_fontsize=None, ylbl_fontsize=None, title_fontsize=None,
			tick_major_fontsize=None, tick_minor_fontsize=None,
			projection=None, zscl='linear', zlbl='z', zlbl_fontsize=None):
    if(xlbl_fontsize is None):
        xlbl_fontsize = lbl_fontsize_dict[font_mode]
    if(ylbl_fontsize is None):
        ylbl_fontsize = lbl_fontsize_dict[font_mode]
    if(title_fontsize is None):
        title_fontsize = lbl_fontsize_dict[font_mode]
    if(tick_major_fontsize is None):
        tick_major_fontsize = ticks_fontsize_dict[font_mode]

    fig = plt.figure()
    fig_num = plt.gcf().number
    ax = fig.add_subplot(projection=projection)

    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlbl, fontsize=xlbl_fontsize)
    plt.ylabel(ylbl, fontsize=ylbl_fontsize)
    plt.xscale(xscl)
    plt.yscale(yscl)
    ax.tick_params(axis='both', which='major', labelsize=tick_major_fontsize)
    if(tick_minor_fontsize is not None):
        ax.tick_params(axis='both', which='minor', labelsize=tick_minor_fontsize)
    return fig, ax, fig_num

# HH neuron parameters
V_rest = -60 # mV, resting membrane potential
V_th = 0 # mV, threshold potential

E_k = -77 # mV, equilibrium potassium potential
E_Na = 50  # mV, equilibrium sodium potential
E_L = -54.4  # mV, equilibrium leakage potential

g_K = 36  # mSm, membrane conductivity for potassium current
g_Na = 120  # mSm, membrane conductivity for sodium current
g_L = 0.3  # mSm, membrane conductivity for leakage current

C = 1  # microF, membrane capacitance

# parameters of the exciting synapse:
tau_rec = 800 # ms, recovery time constant
tau_i = 3 # ms, inactivation time constant
U = 0.5 # fraction of released synaptic resource
A = 350 #pA, variable proportional to synaptic current

# initial values for gate HH neuron variables
h_0 = 0.6
n_0 = 0.3
m_0 = 0.05

# initial values for synaptic resource fractions in short-term plasticity model
x_0 = 1 # recovered state
y_0 = 0 # active state
z_0 = 0 # inactive state
u_0 = U # fraction of released synaptic resource

V_0 = V_rest # initial membrane potential
F_0 = 1 # starting relative spiking phase

I_pre = 20  # microA, input current onto presynaptic neuron

dt = 0.01  # ms, simulation step
start = 0  # ms, start time
stop = 600  # ms, end time
TIME = np.linspace(start, stop, int(stop / dt)) # array of simulation times

V_pre_list = [] # for recording of presynaptic neuron potential dynamics
V_post_list = [] # for recording of postsynaptic neuron potential dynamics

pre_spikes = []  # array for presynaptic spike timing recording
post_spikes = []  # array for postsynaptic spike timing recording

# initial parameters values for presynaptic neuron
V_pre = V_0
h_pre = h_0
n_pre = n_0
m_pre = m_0

#initial parameters values for postsynaptic neuron
V_post = V_0
h_post = h_0
n_post = n_0
m_post = m_0

#initial values for exciting synapse:
x = x_0
y = y_0
z = z_0
u = u_0
F = F_0

F_list = np.zeros(len(TIME)) # relative phase curve array

for i in range(len(TIME)):
    F_old = F
    t = TIME[i]
    # Euler scheme step calculation
    V_pre_old = V_pre
    dV_pre = (I_pre - g_K*(n_pre**4)*(V_pre - E_k) - g_Na*(m_pre**3)*h_pre*(V_pre - E_Na) - g_L*(V_pre - E_L))*dt/C
    dn_pre = (a_n(V_pre)*(1 - n_pre) - b_n(V_pre)*n_pre)*dt
    dm_pre = (a_m(V_pre)*(1 - m_pre) - b_m(V_pre)*m_pre) * dt
    dh_pre = (a_h(V_pre)*(1 - h_pre) - b_h(V_pre)*h_pre) * dt
    V_pre += dV_pre
    n_pre += dn_pre
    m_pre += dm_pre
    h_pre += dh_pre
    V_pre_list.append(V_pre)

    # presynaptic spike checking
    if V_pre >= V_th and V_pre_old < V_th and dV_pre > 0:
        pre_spikes.append(t)

    x_old = x
    y_old = y
    z_old = z
    # last presynaptic spike occuring time
    t_sp_last = max(list(filter(lambda x: x <= t, pre_spikes)), default=0)
    # changes in synapse
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

    I_syn = A*y # synaptic current due to synaptic connection between neurons

    # changes in postsynaptic neuron
    dV_post = (I_syn - g_K*(n_post**4)*(V_post - E_k) - g_Na*(m_post**3)*h_post*(V_post - E_Na) - g_L*(V_post - E_L))*dt/C
    dn_post = (a_n(V_post) * (1 - n_post) - b_n(V_post) * n_post)*dt
    dm_post = (a_m(V_post) * (1 - m_post) - b_m(V_post) * m_post) * dt
    dh_post = (a_h(V_post) * (1 - h_post) - b_h(V_post) * h_post)*dt
    V_post += dV_post
    n_post += dn_post
    m_post += dm_post
    h_post += dh_post
    V_post_list.append(V_post)

    # presynaptic spike checking
    if V_post >= V_th and V_post_list[-2] < V_th:
        post_spikes.append(t)
        F = (t - t_sp_last) # spike timing calculation
    else:
        F = F_old # if postsynapse isn't spiking, phase doesn't change
    F_list[i] = F

T_pre = pre_spikes[-1] - pre_spikes[-2] # presynapse's spiking period calculation
phase_list = [(x/T_pre) for x in F_list]  # relative spiking phase array

font_mode = 'present'

fig, ax, fig_ID = get_fig("time, [ms]", "relative spiking phase")
ax.plot(TIME, phase_list, label="Relative spiking phase Ф(t)", c=(0.01, 0.01, 0.8), lw=2)
plt.tight_layout()
plt.show()