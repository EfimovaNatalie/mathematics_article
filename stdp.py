import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Modeling of Spike-timing dependent plasticity (STDP) for Hodgkin-Huxley (HH) neuron

# definition of gate variables for HH neuron
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

    #fig = plt.figure() if(fig_num is None) else plt.figure(fig_num)
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
E_K = -77 # mV, equilibrium potassium potential
E_Na = 50  # mV, equilibrium sodium potential
E_L = -54.4  # mV, equilibrium leakage potential

g_K = 36  # mSm, membrane conductivity for potassium current
g_Na = 120  # mSm, membrane conductivity for sodium current
g_L = 0.3  # mSm, membrane conductivity for leakage current

V_th = 0 # mV, threshold potential
g_rest = g_K + g_Na + g_L # resting membrane conductivity
C = 1  # microF, membrane capacitance

E_rev = 0 # mV, synapse reversal potential

# parameters of the synapse with STDP
A_plus = 0.001 # magnitude of long-term potentiation
A_minus = A_plus/5  # magnitude of long-term depression
tau_stdp = 50 # ms, STDP time constant

# initial values
g_0 = 0.145 # mSm, initial synaptic conductivity
# initial values for gate HH neuron variables
h_0 = 0.6
n_0 = 0.3
m_0 = 0.05

I_pre = 20 # microA, input current onto presynaptic neuron

dt = 0.01  # ms, simulation step
start = 0  # ms, start time
stop = 5000  # ms, end time
TIME = np.linspace(start, stop, int(stop / dt)) # array of simulation times

V_pre_list = [] # for recording of presynaptic neuron potential dynamics
V_post_list = [] # for recording of postsynaptic neuron potential dynamics

pre_spikes = [] # for presynaptic spike timing recording
post_spikes = [] # for postsynaptic timing recording

X_list = [] # for presynaptic trace recording
Y_list = [] # for postsynaptic trace recording

# initial values for presynaptic neuron
h_pre = h_0
n_pre = n_0
m_pre = m_0

V_rest = (g_K*(n_pre**4)*E_K + g_Na*(m_pre**3)*h_pre*E_Na + g_L*E_L)/(g_L + g_Na*(m_pre**3)*h_pre + g_K*(n_pre**4)) # mV, resting membrane potential
V_0 = V_rest # initial potential value
V_pre = V_0 # initial presynaptic membrane potential

#initial values for postsynaptic neuron
V_post = V_0
h_post = h_0
n_post = n_0
m_post = m_0
g = g_0

x = 0 # postsynaptic trace
y = 0 # presynaptic trace

for t in TIME:
    V_pre_old = V_pre
    V_post_old = V_post
    g_old = g
    x_old = x
    y_old = y
    # Euler scheme step calculation
    dV_pre = (I_pre - g_K*(n_pre**4)*(V_pre - E_K) - g_Na*(m_pre**3)*h_pre*(V_pre - E_Na) - g_L*(V_pre - E_L))*dt/C
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

    I_syn = -g_old * (V_post_old - E_rev)  # calculation of the synaptic current due to synaptic connection between neurons

    # changes in postsynaptic neuron
    dV_post = (I_syn - g_K * (n_post ** 4) * (V_post - E_K) - g_Na * (m_post ** 3) * h_post * (V_post - E_Na) - g_L * (V_post - E_L)) * dt / C
    dn_post = (a_n(V_post) * (1 - n_post) - b_n(V_post) * n_post) * dt
    dm_post = (a_m(V_post) * (1 - m_post) - b_m(V_post) * m_post) * dt
    dh_post = (a_h(V_post) * (1 - h_post) - b_h(V_post) * h_post) * dt
    V_post += dV_post
    n_post += dn_post
    m_post += dm_post
    h_post += dh_post
    V_post_list.append(V_post)

    # postsynaptic spike checking
    if V_post >= V_th and V_post_old < V_th and dV_pre > 0:
        post_spikes.append(t)

    # changes in synapse
    t_pre_sp_last = max(list(filter(lambda x: x <= t, pre_spikes)), default=0)
    t_post_sp_last = max(list(filter(lambda x: x <= t, post_spikes)), default=0)
    delta_t = t_post_sp_last - t_pre_sp_last

    # STDP synaptic changes depending on the sign of delta_t
    if delta_t > 0:
        dx = -x_old * dt / tau_stdp
        if abs(t_post_sp_last - t) <= dt:
            dg = A_plus*x_old
            dy = -y_old*dt/tau_stdp + 1
        else:
            dg = 0
            dy = -y_old*dt/tau_stdp
    elif delta_t < 0:
        dy = -y_old * dt / tau_stdp
        if abs(t_pre_sp_last - t) <= dt:
            dg = - A_minus*y_old
            dx = -x_old*dt/tau_stdp + 1
        else:
            dg = 0
            dx = -x_old * dt / tau_stdp
    else:
        dx = -x_old * dt / tau_stdp
        dy = -y_old * dt / tau_stdp
        dg = 0

    g = g_old + dg
    x = x_old + dx
    y = y_old + dy
    X_list.append(x)
    Y_list.append(y)

fourier_pre = np.fft.fft(V_pre_list[:])/len(V_pre_list[:]) # presynaptic signal fourier transform calculation
fourier_post = np.fft.fft(V_post_list[:])/len(V_post_list[:]) # postsynaptic signal fourier transform calculation
freq = np.fft.fftfreq(np.array(V_post_list[:]).size, d=dt) # frequency axis scale calculating
freq *= 1000 # from mHz to Hz

# plot the membrane potentials dynamics of both neurons
font_mode = 'present'
fig_1, ax_1, fig_ID_1 = get_fig("time, [ms]", "membrane potential, [mV]")
ax_1.plot(TIME, V_pre_list, label='presynaptic neuron', c=(0.01, 0.01, 0.8), lw=2)
ax_1.plot(TIME, V_post_list, label='postsynaptic neuron', c=(1, 0.55, 0), lw=2)
plt.tight_layout()

fig, ax, fig_ID = get_fig("frequency, [Hz]", "fourier transform")
ax.plot(freq, np.abs(fourier_post), label=r"$for V_{post} signal$", c=(1, 0.55, 0), lw=2)
ax.plot(freq, np.abs(fourier_pre), label=r"$for V_{pre} signal$", c=(0.01, 0.01, 0.8), lw=2, linestyle='dotted')
ax.set_ylim(-1, 14)
ax.set_xlim(1.5, 120)

# plot the presynaptic and postsynaptic traces
fig_2, ax_2, fig_ID_2 = get_fig("time, [ms]", "trace")
ax_2.plot(TIME, X_list, label = 'presynaptic trace', c=(0.01, 0.01, 0.8), lw=2)
ax_2.plot(TIME, Y_list, label = 'postsynaptic trace', c=(1, 0.55, 0), lw=2)
plt.tight_layout()
plt.show()