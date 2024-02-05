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

# definition of STDP-curve defining z-plasticity mechanism
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

#parameters of z-plasticity model
I_pre = 0.8 # pA, presynaptic current defining z_pre
I_post = 0.8 # pA, postsynaptic current defining z_post
g_syn = 0.22 # mSm, synaptic conductivity

I_input = 11 # pA, input current for the presynaptic neuron

#parameters for presynaptic feedback regulation
F_c = 0.2 # reference ("control") phase
alpha_pre = 0.08 # timescale of the presynaptic polarization’s relaxation

# HH neuron parameters
V_rest = -60  # mV, resting membrane potential
V_th = 0  # mV, threshold potential

E_k = -77  # mV, equilibrium potassium potential
E_Na = 50  # mV, equilibrium sodium potential
E_L = -54.4  # mV, equilibrium leakage potential

E_rev = 0 # mV, reversion potential of the synapse

g_K = 36  # mSm, membrane conductivity for potassium current
g_Na = 120  # mSm, membrane conductivity for sodium current
g_L = 0.3  # mSm, membrane conductivity for leakage current

C = 1  # microF, membrane capacitance

# initial values for gate HH neuron variables
h_0 = 0.6
n_0 = 0.3
m_0 = 0.05

V_0 = V_rest # initial membrane potential

F_0 = 1 # starting relative spiking phase value

# initial values for z-plasticity parameters
z_pre_0 = I_pre # initial z-variable for presynaptic neuron
z_post_0 = I_post # initial z-variable for postsynaptic neuron

dt = 0.01  # ms, simulation step
t_start = 0  # ms, start time
t_stop = 3000  # ms, end time
TIME = np.linspace(t_start, t_stop, int(t_stop / dt)) # array of simulation times

t_k = 1800 # time moment of turning on z-stdp mechanism when k_pre != 0
step_t_k = int(t_k/dt) # time step of turning on z-stdp mechanism

# setting initial values to variables
F = F_0 # relative phase
z_pre = z_pre_0 # presynaptic neuron’s state parameter
z_post = z_post_0 # postsynaptic neuron’s state parameter

T_pre = t_stop # presynaptic neuron spiking period

# initial values for presynaptic neuron
V_pre = V_0
h_pre = h_0
n_pre = n_0
m_pre = m_0

#initial values for postsynaptic neuron
V_post = V_0
h_post = h_0
n_post = n_0
m_post = m_0

V_pre_list = [] # array for presynaptic neuron potential dynamics recording
V_post_list = [] # array for postsynaptic neuron potential dynamics recording

F_list = [] # array for relative spiking phase recording
F_c_list = np.ones(len(TIME))*(F_c) # control phase array

z_pre_list = [] # array for z_pre variable dynamics recording
t_pre_spikes = [] # array for presynaptic spike timing recording

V_pre_list.append(V_pre)
V_post_list.append(V_post)
F_list.append(F)
z_pre_list.append(z_pre)

for t in TIME[1:-step_t_k]:
    k_pre = 0 # z-plasticity mechanism isn't working by the moment of t_k
    F = F_list[-1]
    V_pre_old = V_pre
    V_post_old = V_post
    z_pre_old = z_pre
    # Euler scheme step calculation for presynapse
    dV_pre = (I_input - g_K*(n_pre**4)*(V_pre_old - E_k) - g_Na*(m_pre**3)*h_pre*(V_pre_old - E_Na) - g_L*(V_pre_old - E_L)
               - z_pre)*dt/C
    dn_pre = (a_n(V_pre_old) * (1 - n_pre) - b_n(V_pre_old) * n_pre) * dt
    dm_pre = (a_m(V_pre_old) * (1 - m_pre) - b_m(V_pre_old) * m_pre) * dt
    dh_pre = (a_h(V_pre_old) * (1 - h_pre) - b_h(V_pre_old) * h_pre) * dt
    dz_pre = (alpha_pre * (I_pre - z_pre_old) + k_pre * G(F)) * dt
    V_pre = V_pre_old + dV_pre
    n_pre += dn_pre
    m_pre += dm_pre
    h_pre += dh_pre
    z_pre = z_pre_old + dz_pre
    V_pre_list.append(V_pre)
    z_pre_list.append(z_pre)

    I_syn = (E_rev - V_post_old)*g_syn # synaptic current calculation

    # Euler scheme step calculation for postsynapse
    dV_post = (I_syn - g_K*(n_post**4)*(V_post_old - E_k) - g_Na*(m_post**3)*h_post*(V_post_old - E_Na) - g_L*(V_post_old - E_L)
                - z_post)*dt/C
    dn_post = (a_n(V_post_old) * (1 - n_post) - b_n(V_post_old) * n_post) * dt
    dm_post = (a_m(V_post_old) * (1 - m_post) - b_m(V_post_old) * m_post) * dt
    dh_post = (a_h(V_post_old) * (1 - h_post) - b_h(V_post_old) * h_post) * dt
    n_post += dn_post
    m_post += dm_post
    h_post += dh_post
    V_post = V_post_old + dV_post
    V_post_list.append(V_post)

    # presynaptic spike checking
    if V_pre >= V_th and V_pre_old < V_th:
        t_pre_spikes.append(t)
    # postsynaptic spike checking
    if V_post >= V_th and V_post_old < V_th:
        t_pre_last = max(filter(lambda x: x < t, t_pre_spikes), default=0)
        F = t - t_pre_last # spike timing calculation
        F_list.append(F)
    else:
        F_list.append(F) # if postsynapse isn't spiking, phase doesn't change

if len(t_pre_spikes) > 2:
    periods = [t_pre_spikes[i+1] - t_pre_spikes[i] for i in range(len(t_pre_spikes) - 1)]
    T_pre = sum(periods)/len(periods) # presynapse's spiking period calculation
phase_list = [x/T_pre for x in F_list] # relative spiking phase array

for t in TIME[-step_t_k::]:
    k_pre = 0.15 # now z-plasticity mechanism is turned on
    F = phase_list[-1]
    V_pre_old = V_pre
    V_post_old = V_post
    z_pre_old = z_pre
    # Euler scheme step calculation for presynapse
    dV_pre = (I_input - g_K*(n_pre**4)*(V_pre_old - E_k) - g_Na*(m_pre**3)*h_pre*(V_pre_old - E_Na) - g_L*(V_pre_old - E_L)
               - z_pre)*dt/C
    dn_pre = (a_n(V_pre_old) * (1 - n_pre) - b_n(V_pre_old) * n_pre) * dt
    dm_pre = (a_m(V_pre_old) * (1 - m_pre) - b_m(V_pre_old) * m_pre) * dt
    dh_pre = (a_h(V_pre_old) * (1 - h_pre) - b_h(V_pre_old) * h_pre) * dt
    dz_pre = (alpha_pre * (I_pre - z_pre_old) + k_pre * G(F)) * dt
    V_pre = V_pre_old + dV_pre
    n_pre += dn_pre
    m_pre += dm_pre
    h_pre += dh_pre
    z_pre = z_pre_old + dz_pre
    V_pre_list.append(V_pre)
    z_pre_list.append(z_pre)

    I_syn = (E_rev - V_post_old) * g_syn # synaptic current

    # Euler scheme step calculation for postsynapse
    dV_post = (I_syn - g_K*(n_post**4)*(V_post_old - E_k) - g_Na*(m_post**3)*h_post*(V_post_old - E_Na) - g_L*(V_post_old - E_L)
                - z_post)*dt/C
    dn_post = (a_n(V_post_old) * (1 - n_post) - b_n(V_post_old) * n_post) * dt
    dm_post = (a_m(V_post_old) * (1 - m_post) - b_m(V_post_old) * m_post) * dt
    dh_post = (a_h(V_post_old) * (1 - h_post) - b_h(V_post_old) * h_post) * dt
    n_post += dn_post
    m_post += dm_post
    h_post += dh_post
    V_post = V_post_old + dV_post
    V_post_list.append(V_post)

    # presynaptic spike checking
    if V_pre >= V_th and V_pre_old < V_th:
        t_pre_spikes.append(t)
    # postsynaptic spike checking
    if V_post >= V_th and V_post_old < V_th:
        t_pre_last = max(list(filter(lambda x: x < t, t_pre_spikes)), default=0)
        if len(t_pre_spikes) > 2:
            T_pre = t_pre_last - t_pre_spikes[-2] # presynapse spiking period calculation
        else:
            T_pre = t - t_pre_last
        F = (t - t_pre_last)/T_pre # relative spiking phase
        phase_list.append(F)
    else:
        F_last = phase_list[-1] # if postsynapse isn't spiking, phase doesn't change
        phase_list.append(F_last)

# Fourier transform calculation for pre- and postsynaptic signals
fourier_pre = np.fft.fft(V_pre_list[130000:])/len(V_pre_list[130000:])
fourier_post = np.fft.fft(V_post_list[130000:])/len(V_post_list[130000:])
freq = np.fft.fftfreq(np.array(V_post_list[130000:]).size, d=dt) # frequency axis scale
freq *= 1000

font_mode = 'present'

# plot the fourier transform graphs
fig_1, ax_1, fig_ID_1 = get_fig("frequency, [Hz]", "fourier transform")
ax_1.plot(freq, np.abs(fourier_post), label=r"$for V_{post} signal$", c=(1, 0.55, 0), lw=2)
ax_1.plot(freq, np.abs(fourier_pre), label=r"$for V_{pre} signal$", c=(0.01, 0.01, 0.8), lw=3, linestyle='dotted')
ax_1.set_xlim(0.8, 100)
ax_1.set_ylim(-1, 11)
plt.tight_layout()
# plot z_pre variable dynamics
fig_2, ax_2, fig_ID_2 = get_fig("time, [ms]", "$I^{extra}_{pre}(t)$")
ax_2.plot(TIME, z_pre_list, label=r"$I^{extra}_{pre}(t)$", c=(0.01, 0.01, 0.8), lw=2)
#ax_2.set_ylim(0.7, 2.7)
plt.tight_layout()
# plot relative spiking phase dynamics in comparison with reference ("control") phase F_с
fig_3, ax_3, fig_ID_3 = get_fig("time, [ms]", "relative spiking phase") #$\Phi$")
ax_3.plot(TIME, phase_list, label="$\Phi(t)$", c=(0.01, 0.01, 0.8), lw=2)
ax_3.plot(TIME, F_c_list, label="$\Phi_{c}$", c=(1, 0.55, 0), lw=2)
plt.tight_layout()
# plot membrane potentials dynamics for both neurons
fig_4, ax_4, fig_ID_4 = get_fig("time, [ms]", "membrane potential, [mV]")
ax_4.plot(TIME, V_pre_list, label=r"$V_{pre}(t)$", c=(0.01, 0.01, 0.8), lw=2)
ax_4.plot(TIME, V_post_list, label=r"$V_{post}(t)$", c=(1, 0.55, 0), lw=2)
ax_4.set_ylim(-85, 45)
plt.tight_layout()
plt.show()