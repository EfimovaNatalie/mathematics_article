import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

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

# HH neuron parameters
V_rest = -60 #mV
V_th = -50 #mV

E_k = -77 # mV
E_Na = 50  # mV
E_L = -54.4  # mV

g_K = 36  #mSm
g_Na = 120  # mSm
g_L = 0.3  # mSm

C = 1 # microF
g_max = 30

# parameters for exciting synapse:
tau_rec = 800 #ms
tau_i = 3 #ms
U = 0.5

# initial values
h_0 = 0.6
n_0 = 0.3
m_0 = 0.05
x_0 = 1
y_0 = z_0 = 0
u_0 = U
V_0 = V_rest

#STDP synapse parameters
A_plus = 0.1 # magnitude of LTP
A_minus = -A_plus  # magnitude of LTD
#print(A_minus)
tau_stdp = 20 # STDP time constant [ms]
g_0 = g_max #g_rest #0.005*g_rest
E_rev = 0 #mV

I_pre = 30 # microA

dt = 0.01  # ms
start = 0  # ms
stop = 5000  # ms
TIME = np.linspace(start, stop, int(stop / dt))

V_pre_list = [] # for recording of presynaptic neuron potential dynamics
V_post_list = []
pre_spikes = [] # for presynaptic spike timing recording
post_spikes = [] # for postsynaptic timing record
#lists for synaptic resources dynamic recording
X_list = []
Y_list = []
Z_list = []

I_list = []
x_trace_list = []
y_trace_list = []
G_list = []
PRE_spikes = np.zeros(len(TIME))
POST_spikes = np.zeros(len(TIME))

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

#initial values for exciting synapse:
A = 300 #variable which is proportional with synaptic current
x = x_0
y = y_0
z = z_0
u = u_0
g = g_0

x_trace = 0 #postsynaptic trace
y_trace = 0 #presynaptic trace

interval = 500 #ms
num_intervals = int(stop/interval)
frequencies_per_interval = []
intervals = []
indexes_of_times = [int(n*interval/dt) for n in range(num_intervals)]
num_spikes_per_interval = 0

for i in range(len(TIME)):
    t = TIME[i]
    V_pre_old = V_pre
    V_post_old = V_post
    x_old = x
    y_old = y
    z_old = z
    g_old = g
    x_old_trace = x_trace
    y_old_trace = y_trace

    dV_pre = (I_pre - g_K*(n_pre**4)*(V_pre_old - E_k) - g_Na*(m_pre**3)*h_pre*(V_pre_old - E_Na) - g_L*(V_pre_old - E_L))*dt/C
    dn_pre = (a_n(V_pre_old)*(1 - n_pre) - b_n(V_pre_old)*n_pre)*dt
    dm_pre = (a_m(V_pre_old)*(1 - m_pre) - b_m(V_pre_old)*m_pre) * dt
    dh_pre = (a_h(V_pre_old)*(1 - h_pre) - b_h(V_pre_old)*h_pre) * dt

    V_pre = V_pre_old + dV_pre
    n_pre += dn_pre
    m_pre += dm_pre
    h_pre += dh_pre

    if V_pre >= V_th and V_pre_old < V_th and dV_pre > 0:
        pre_spikes.append(t)
        PRE_spikes[i] = 1
    V_pre_list.append(V_pre)

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

    #I_syn = A*y # synaptic current due to synaptic connection between neurons
    I_syn = g_old*y*(E_rev - V_post_old)  # synaptic current due to synaptic connection between neurons

    # changes in postsynaptic neuron
    dV_post = (I_syn - g_K*(n_post**4)*(V_post_old - E_k) - g_Na*(m_post**3)*h_post*(V_post_old - E_Na) - g_L*(V_post_old - E_L))*dt/C
    dn_post = (a_n(V_post_old) * (1 - n_post) - b_n(V_post_old) * n_post)*dt
    dm_post = (a_m(V_post_old) * (1 - m_post) - b_m(V_post_old) * m_post) * dt
    dh_post = (a_h(V_post_old) * (1 - h_post) - b_h(V_post_old) * h_post)*dt

    V_post = V_post_old + dV_post
    n_post += dn_post
    m_post += dm_post
    h_post += dh_post

    if V_post >= V_th and V_post_old < V_th:
        post_spikes.append(t)
        POST_spikes[i] = 1
        num_spikes_per_interval += 1
    V_post_list.append(V_post)

    if i in indexes_of_times:
        print(t, num_spikes_per_interval)
        frequency_per_interval = 1000*num_spikes_per_interval/interval #Hz
        frequencies_per_interval.append(frequency_per_interval)
        intervals.append(t)
        num_spikes_per_interval = 0

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

    g = g_old + dg
    x_trace = x_old_trace + dx_trace
    y_trace = y_old_trace + dy_trace
    x_trace_list.append(x_trace)
    y_trace_list.append(y_trace)
    G_list.append(g)

# Fourier transform calculation for pre- and postsynaptic signals
fourier_post = np.fft.fft(V_post_list)/len(V_post_list)
fourier_pre = np.fft.fft(V_pre_list)/len(V_pre_list)
freq = np.fft.fftfreq(np.array(V_post_list).size, d=dt)
freq *= 1000

font_mode = 'present'

fig_1, ax_1, fig_ID_1 = get_fig("time, [ms]", "membrane potential, [mV]")
ax_1.plot(TIME, V_pre_list, label=r"$V_{pre}(t)$", c=(0.01, 0.01, 0.8), lw=1)
ax_1.plot(TIME, V_post_list, label=r"$V_{post}(t)$", c=(1, 0.55, 0), lw=1)
ax_1.set_ylim(-85, 55)
plt.tight_layout()

fig_4, ax_4, fig_ID_4 = get_fig("frequency, [Hz]", "fourier transform")
ax_4.plot(freq, np.abs(fourier_post), label=r"$for V_{post} signal$", c=(1, 0.55, 0), lw=2)
ax_4.plot(freq, np.abs(fourier_pre), label=r"$for V_{pre} signal$", c=(0.01, 0.01, 0.8), lw=3, linestyle='dotted')
ax_4.set_xlim(2, 123)
ax_4.set_ylim(-1, 16)
plt.tight_layout()

#plt.legend()
'''plt.figure(figsize=(10, 10))
plt.plot(TIME, x_trace_list, label = 'presynaptic trace')
plt.plot(TIME, y_trace_list, label = 'postsynaptic trace')
plt.plot(TIME, G_list, label = 'synaptic conductance')
plt.plot(TIME, PRE_spikes, label='pre_spikes')
plt.plot(TIME, POST_spikes, label='post_spikes')
plt.legend()
plt.figure(figsize=(10,14))
plt.scatter(intervals, frequencies_per_interval, label="frequencies per interval", s=20, c="red")
plt.xlabel("time, [ms]")
plt.ylabel("firing rate, [Hz]")
mpl.rcParams['font.size'] = 20
plt.legend()'''
plt.show()