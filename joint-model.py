import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Modeling of joint model of short-term (STP), spike-timing dependent (STDP) and z-plasticity for Hodgkin-Huxley (HH) neuron

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
def G(x):
    return np.sin(2*np.pi*(x - F_c))

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
E_k = -77 # mV, equilibrium potassium potential
E_Na = 50  # mV, equilibrium sodium potential
E_L = -54.4  # mV, equilibrium leakage potential

g_K = 36  # mSm, membrane conductivity for potassium current
g_Na = 120  # mSm, membrane conductivity for sodium current
g_L = 0.3  # mSm, membrane conductivity for leakage current

C = 1  # microF, membrane capacitance

V_rest = -60  # mV, resting membrane potential
V_th = 0  # mV, threshold potential

g_m = 10 # mSm, membrane conductivity

#parameters of z-plasticity model
I_pre = 0.7 # pA, presynaptic current defining z_pre
I_post = 0.7 # pA, postsynaptic current defining z_post
I = 14 # pA, input current for the presynaptic neuron

#parameters for presynaptic feedback regulation
F_c = 0.5 # reference ("control") phase
alpha_pre = 0.0015 # timescale of the presynaptic polarization’s relaxation

# parameters of the exciting synapse:
tau_rec = 800 # ms, recovery time constant
tau_i = 3 # ms, inactivation time constant
U = 0.5 # fraction of released synaptic resource

# initial values for gate HH neuron variables
h_0 = 0.6
n_0 = 0.3
m_0 = 0.05

V_0 = V_rest # initial membrane potential

F_0 = 1 # starting relative spiking phase value

# initial values for synaptic resource fractions in short-term plasticity model
x_0 = 1 # recovered state
y_0 = 0 # active state
z_0 = 0 # inactive state
u_0 = U # fraction of released synaptic resource

# initial values for z-plasticity parameters
z_pre_0 = I_pre # initial z-variable for presynaptic neuron
z_post_0 = I_post # initial z-variable for postsynaptic neuron

# parameters of the synapse with STDP
A_plus = 0.1 # magnitude of long-term potentiation
A_minus = -A_plus  # magnitude of long-term depression
tau_stdp = 20  # ms, STDP time constant
g_0 = g_m # initial synapse conductivity
E_rev = 0  # mV, reversal synaptic potential

dt = 0.01  # ms, simulation step
t_start = 0  # ms, start time
t_stop = 2500  # ms, end time
TIME = np.linspace(t_start, t_stop, int(t_stop / dt)) # array of simulation times

t_k = t_stop-700 # ms, moment of turning on of z-plasticity mechanism
step_t_k = int(t_k/dt) # time step of the t_k moment

V_pre_list = [] # for recording of presynaptic neuron potential dynamics
V_post_list = [] # for recording of postsynaptic neuron potential dynamics

pre_spikes = []  # for presynaptic spike timing recording
post_spikes = []  # for postsynaptic spike timing recording

x_trace_list = [] # for presynaptic trace (in STDP model) recording
y_trace_list = [] # for postsynaptic trace (in STDP model) recording
G_list = [] # for the synaptic conductivity dynamics recording

PRE_spikes = np.zeros(len(TIME)) # list of the presynaptic spikes moments
POST_spikes = np.zeros(len(TIME)) # list of the postsynaptic spikes moments

# setting initial values for presynaptic neuron variables
V_pre = V_0
h_pre = h_0
n_pre = n_0
m_pre = m_0

# setting initial values for postsynaptic neuron variables
V_post = V_0
h_post = h_0
n_post = n_0
m_post = m_0

# setting initial values for other variables
F = F_0 # initial relative spiking phase
z_pre = z_pre_0
z_post = z_post_0

T_pre = t_stop # presynaptic neuron spiking period

# initial values of the exciting synapse variables:
x = x_0
y = y_0
z = z_0
u = u_0
g = g_0 # initial value of the synaptic conductivity

x_trace = 0  # presynaptic trace
y_trace = 0  # postsynaptic trace

F_list = [] # list for relative spike timing recording
F_c_list = np.ones(len(TIME)) * (F_c) # control phase array
z_pre_list = [] # list for z_pre variable dynamics recording

V_pre_list.append(V_pre)
V_post_list.append(V_post)
F_list.append(F)
z_pre_list.append(z_pre)
x_trace_list.append(x_trace)
y_trace_list.append(y_trace)
G_list.append(g)

for t in TIME[1:-step_t_k]:
    k_pre = 0 # z-plasticity mechanism isn't working by the moment of t_k
    # HH neurons variables
    V_pre_old = V_pre
    V_post_old = V_post
    # stp variables
    x_old = x
    y_old = y
    z_old = z
    # stdp variables
    g_old = g
    x_old_trace = x_trace
    y_old_trace = y_trace
    # z-plasticity variables
    F = F_list[-1]
    z_pre_old = z_pre

    # Euler scheme step calculation for presynapse
    dV_pre = (I - g_K*(n_pre**4)*(V_pre_old - E_k) - g_Na*(m_pre**3)*h_pre*(V_pre_old - E_Na) - g_L*(V_pre_old - E_L)
               - z_pre)*dt/C
    dn_pre = (a_n(V_pre_old) * (1 - n_pre) - b_n(V_pre_old) * n_pre) * dt
    dm_pre = (a_m(V_pre_old) * (1 - m_pre) - b_m(V_pre_old) * m_pre) * dt
    dh_pre = (a_h(V_pre_old) * (1 - h_pre) - b_h(V_pre_old) * h_pre) * dt
    V_pre = V_pre_old + dV_pre
    n_pre += dn_pre
    m_pre += dm_pre
    h_pre += dh_pre

    # presynaptic spike checking
    if V_pre >= V_th and V_pre_old < V_th and dV_pre > 0:
        pre_spikes.append(t)
    # changes in synapse
    t_sp_last = max(list(filter(lambda x: x <= t, pre_spikes)), default=0)
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

    I_syn = g_old*y_old*(E_rev - V_post_old)  # synaptic current due to synaptic connection between neurons

    # Euler scheme step calculation for presynapse
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

    # STDP synaptic changes depending on the sign of delta_t
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
    # changes in synapse due to STDP mechanism
    g = g_old + dg
    x_trace = x_old_trace + dx_trace
    y_trace = y_old_trace + dy_trace

    V_pre_list.append(V_pre)
    V_post_list.append(V_post)
    z_pre_list.append(z_pre)
    x_trace_list.append(x_trace)
    y_trace_list.append(y_trace)
    G_list.append(g)

    # chicking for the postsymaptic spike
    if V_post >= V_th and V_post_old < V_th:
        t_pre_last = max(filter(lambda x: x < t, pre_spikes), default=0)
        F = t - t_pre_last
        F_list.append(F)
    else:
        F_list.append(F)

if len(pre_spikes) > 2:
    periods = [pre_spikes[i+1] - pre_spikes[i] for i in range(len(pre_spikes) - 1)]
    T_pre = sum(periods)/len(periods) # presynaptic neuron spiking period calculation
F_list_new = [x/T_pre for x in F_list] # array of the relative spiking phase between two neurons

for t in TIME[-step_t_k::]:
    k_pre = 0.5 # now z-plasticity mechanism is turned on
    # HH neurons variables
    V_pre_old = V_pre
    V_post_old = V_post
    # stp variables
    x_old = x
    y_old = y
    z_old = z
    # stdp variables
    g_old = g
    x_old_trace = x_trace
    y_old_trace = y_trace
    # z-plasticity variables
    F = F_list_new[-1]
    z_pre_old = z_pre

    # Euler scheme step claculation for the presynapse
    dV_pre = (I - g_K*(n_pre**4)*(V_pre_old - E_k) - g_Na*(m_pre**3)*h_pre*(V_pre_old - E_Na) - g_L*(V_pre_old - E_L) - z_pre)*dt/C
    dn_pre = (a_n(V_pre_old) * (1 - n_pre) - b_n(V_pre_old) * n_pre) * dt
    dm_pre = (a_m(V_pre_old) * (1 - m_pre) - b_m(V_pre_old) * m_pre) * dt
    dh_pre = (a_h(V_pre_old) * (1 - h_pre) - b_h(V_pre_old) * h_pre) * dt

    V_pre = V_pre_old + dV_pre
    n_pre += dn_pre
    m_pre += dm_pre
    h_pre += dh_pre

    # presynaptic spike checking
    if V_pre >= V_th and V_pre_old < V_th and dV_pre > 0:
        pre_spikes.append(t)
        #PRE_spikes[i] = 1

    # changes in synapse
    t_sp_last = max(list(filter(lambda x: x <= t, pre_spikes)), default=0)
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

    # Euler scheme step calculation for the postsynapse
    dV_post = (I_syn - g_K*(n_post**4)*(V_post_old - E_k) - g_Na*(m_post**3)*h_post*(V_post_old - E_Na) - g_L*(V_post_old - E_L)
               - z_post) * dt / C
    dn_post = (a_n(V_post_old) * (1 - n_post) - b_n(V_post_old) * n_post) * dt
    dm_post = (a_m(V_post_old) * (1 - m_post) - b_m(V_post_old) * m_post) * dt
    dh_post = (a_h(V_post_old) * (1 - h_post) - b_h(V_post_old) * h_post) * dt

    V_post = V_post_old + dV_post
    n_post += dn_post
    m_post += dm_post
    h_post += dh_post

    # z-plasticity mechanism
    dz_pre = (alpha_pre * (I_pre - z_pre_old) + k_pre * G(F)) * dt
    z_pre = z_pre_old + dz_pre

    # postsynaptic spike checking
    if V_post >= V_th and V_post_old < V_th:
        post_spikes.append(t)
        # POST_spikes[i] = 1

    # changes in synapse
    t_pre_sp_last = max(list(filter(lambda n: n <= t, pre_spikes)), default=0)
    t_post_sp_last = max(list(filter(lambda n: n <= t, post_spikes)), default=0)
    delta = t_post_sp_last - t_pre_sp_last

    # STDP synaptic changes depending on the sign of delta_t
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
    # changes in synapse due to stdp mechanism
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
        F = (t - t_pre_last) / T_pre
        F_list_new.append(F)
    else:
        F_last = F_list_new[-1]
        F_list_new.append(F_last)

# Fourier transform calculation for pre- and postsynaptic signals
fourier_post = np.fft.fft(V_post_list[140000:])/len(V_post_list[140000:])
fourier_pre = np.fft.fft(V_pre_list[140000:])/len(V_pre_list[140000:])
freq = np.fft.fftfreq(np.array(V_post_list[140000:]).size, d=dt)
freq *= 1000

font_mode = 'present'

# plot membrane potentials dynamics for both neurons
"""plt.figure(figsize=(10, 10))
mpl.rcParams['font.size'] = 20
plt.plot(TIME, V_pre_list, label='for presynaptic neuron', c=(0.01, 0.01, 0.8), lw=2)
plt.plot(TIME, V_post_list, label='for postsynaptic neuron', c=(1, 0.55, 0), lw=2)
plt.ylabel("membrane potential, [mV]")
plt.xlabel("time, [ms]")
plt.tight_layout()
plt.legend()"""
fig_1, ax_1, fig_ID_1 = get_fig("time, [ms]", "membrane potential, [mV]")
ax_1.plot(TIME, V_pre_list, label=r"$V_{pre}(t)$", c=(0.01, 0.01, 0.8), lw=2)
ax_1.plot(TIME, V_post_list, label=r"$V_{post}(t)$", c=(1, 0.55, 0), lw=2)
ax_1.set_ylim(-85, 55)
plt.tight_layout()

# plot relative spiking phase dynamics in comparison with reference ("control") phase F_с
"""plt.figure(figsize=(10, 10))
mpl.rcParams['font.size'] = 20
plt.plot(TIME, F_list_new, label="Ф(t)", c=(0.01, 0.01, 0.8), lw=2)
plt.plot(TIME, F_c_list, label="Ф_c", c=(1, 0.55, 0), lw=2)
plt.xlabel("time, [ms]")
plt.ylabel("Ф(t)")
#plt.title("Relative spiking phase")
plt.tight_layout()
plt.legend()"""
fig_2, ax_2, fig_ID_2 = get_fig("time, [ms]", "relative spiking phase") #$\Phi$")
ax_2.plot(TIME, F_list_new, label="$\Phi(t)$", c=(0.01, 0.01, 0.8), lw=2)
ax_2.plot(TIME, F_c_list, label="$\Phi_{c}$", c=(1, 0.55, 0), lw=2)
plt.tight_layout()

# plot z_pre variable dynamics
"""plt.figure(figsize=(10, 10))
mpl.rcParams['font.size'] = 20
plt.plot(TIME, z_pre_list, label=r"$I^{extra}_{pre}(t)$", c=(0.01, 0.01, 0.8), lw=2)
plt.xlabel("time, [ms]")
plt.ylabel("$z_{pre}(t)$")
plt.tight_layout()
plt.legend()"""
fig_3, ax_3, fig_ID_3 = get_fig("time, [ms]", "$I^{extra}_{pre}(t)$")
ax_3.plot(TIME, z_pre_list, label=r"$I^{extra}_{pre}(t)$", c=(0.01, 0.01, 0.8), lw=2)
#ax_2.set_ylim(0.7, 2.7)
plt.tight_layout()

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

# plot the fourier transform graphs
"""plt.figure(figsize=(10, 10))
mpl.rcParams['font.size'] = 20
plt.plot(freq, np.abs(fourier_post), label=r"for $V_{post}$", c=(1, 0.55, 0), lw=2)
plt.plot(freq, np.abs(fourier_pre), label=r"for $V_{pre}$", c=(0.01, 0.01, 0.8), lw=2, linestyle='dotted')
plt.xlabel("frequency, [Hz]")
plt.ylabel("fourier transform")
plt.xlim(0, 500)
plt.ylim(-0.5, 20)
plt.legend()"""
fig_4, ax_4, fig_ID_4 = get_fig("frequency, [Hz]", "fourier transform")
ax_4.plot(freq, np.abs(fourier_post), label=r"$for V_{post} signal$", c=(1, 0.55, 0), lw=2)
ax_4.plot(freq, np.abs(fourier_pre), label=r"$for V_{pre} signal$", c=(0.01, 0.01, 0.8), lw=3, linestyle='dotted')
ax_4.set_xlim(2, 150)
ax_4.set_ylim(-1, 10)
plt.tight_layout()
plt.show()