#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 11:20:49 2021

@author: evbernardes
"""
import os
import time
import numpy as np
from pathlib import Path
from numpy.linalg import norm
import json
import quaternionic as quat

from .helpers import cross, ad, get_middle_vector, mstack
from .helpers import TORAD, Id, zeroM, zeroV, ez, Ez, eps

# get file directory for Blender hack
path = os.path.dirname(os.path.realpath(__file__))

# Helpers for saving and loading
arrlist_quat = ['q', 'q_measured', 'q_drift']

arrlist_1 = [
        't', 'alpha', 'beta', 'drift_angle', 'drift_angle_measured',
        'rotvel', 'pre', 'nut', 'spin', 'alpha_lim', 'fz', 'tz']

arrlist_3 = [
        'nmiddle', 'nmiddle_measured', 'nmiddledot', 'w', 'w_measured',
        'w_delta', 'w_delta_int', 'w_delta_der', 't_ctrl', 'f_ctrl',
        't_grav', 'f_grav', 't_prop', 'f_prop', 't_aero', 't_rndn', 'f_rndn',
        't_imp', 'f_imp', 'Xd', 'angles', 'pos', 'v', 'nd_', 'nd',
        ]


def save_param_dict(sim, datadict, param_dict_name):
    param_dict = getattr(sim, param_dict_name)
    for item in param_dict.items():
        datadict[f'{param_dict_name}_{item[0]}'] = item[1]


def load_param_dict(datadict, param_dict_name):
    param_dict = {}
    L = len(param_dict_name)
    for item in datadict.items():
        if (item[0][:L] == param_dict_name):
            param_dict[item[0][L+1:]] = item[1]
    return param_dict


def set_blender_file(sim,
                     frame_step=1, show_pos=False, frames_max=1000):

    step = int(sim.N / frames_max)

    data = {}
    data['t'] = list(sim.t[::step])
    data['dt'] = sim.DT
    data['tmax'] = sim.tmax
    data['x'] = list(sim.pos.T[0])
    data['y'] = list(sim.pos.T[1])
    data['z'] = list(sim.pos.T[2])
    data['show_pos'] = int(show_pos)
    data['qr'] = list(sim.q.real[::step])
    data['qx'] = list(sim.q.x[::step])
    data['qy'] = list(sim.q.y[::step])
    data['qz'] = list(sim.q.z[::step])
    data['gamma'] = list(sim.angles.T[2][::step])
    data['alpha'] = list(sim.angles.T[1][::step])
    data['beta'] = list(sim.angles.T[0][::step])
    data['frame_step'] = frame_step

    with open(path+'/.blender_temp.json', 'w') as outfile:
        json.dump(data, outfile)


class Monospinner:
    def play(self, frame_step=1, show_pos=False, frames_max=1000):

        set_blender_file(self,
                         frame_step=frame_step,
                         show_pos=show_pos,
                         frames_max=frames_max)

        path_posix = path.replace(' ', '\ ')
        os.system(f'blender --python {path_posix}/blender_test.py')
        os.remove(path+'/.blender_temp.json')

        return path

    def __init__(self, parameters):

        # set all elements
        self.set_parameters(parameters, False)
        self.init_arrays()

    def save(self, filename):
        data = {}
        data['parameters'] = self.parameters
        data['N'] = self.N
        data['i'] = self.i

        for att in arrlist_1+arrlist_3+arrlist_quat:
            data[f'{att}'] = getattr(self, att).tolist()

        dirs = os.path.dirname(filename)

        if dirs != '' and not os.path.exists(dirs):
            os.makedirs(dirs, exist_ok=True)

        with open(filename, 'w') as outfile:
            json.dump(data, outfile)

    @classmethod
    def load(cls, filename):

        with open(filename) as json_file:
            data = json.load(json_file)

        sim = cls(data['parameters'])

        for att in arrlist_1+arrlist_3:
            try:
                setattr(sim, att, np.array(data[f'{att}']))
            except:
                pass

        for att in arrlist_quat:
            try:
                setattr(sim, att, quat.array(data[f'{att}']))
            except:
                pass

        return sim

    def init_arrays(self):

        N = self.N

        # setting impulses
        if self.number_impulses != 0:
            self.impulses_idx = np.linspace(0, self.N-1, self.number_impulses)
            self.impulses_idx = self.impulses_idx.astype(int)
            self.t_imp[self.impulses_idx] = self.noise_impulses[0] * np.vstack([2*np.random.rand(2,self.number_impulses)-1,np.zeros(self.number_impulses)]).T
            self.f_imp[self.impulses_idx] = self.noise_impulses[1] * (2*np.random.rand(3,self.number_impulses).T - 1)

        self.t_rndn = self.noise_random[0] * np.vstack([2*np.random.rand(2,N)-1,np.zeros(N)]).T
        self.f_rndn = self.noise_random[1] * (2*np.random.rand(N,3)-1)

        self.i = 0

        # setting goal arrays
        self.nd = np.zeros([N,3])
        idx = np.array_split(range(N), self.ngoal)
        for i in range(self.ngoal):
            self.nd[idx[i]] = self.nd_[i]

        nmiddledot = (self.nd + ez).T/norm(self.nd + ez, axis=1)
        self.nmiddledot = nmiddledot = nmiddledot.T

        # set initial values
        self.q[0] = self.q0
        self.drift_angle[0] = self.drift_angle_init
        self.q_drift[0] = quat.array.from_axis_angle(self.drift_angle[0] * ez)
        self.q_measured[0] = self.q[0] * self.q_drift[0]
        self.nmiddle_measured[0] = self.nmiddle[0] = get_middle_vector(self.q0)
        self.terminal_wz = self.angvelz_ratio * self.gamma_d
        self.w[0][-1] = self.init_spin_percentage * self.terminal_wz
        self.w_measured[0] = self.w[0]
        self.pos[0] = self.pos0
        self.pre[0], self.nut[0], self.spin[0] = self.q[0].to_euler_angles.T
        self.spin[0] += self.pre[0]

        self.rotvel[0] = self.gamma_d

    # main loop
    def run(self, stop_at_goal=None):

        if stop_at_goal is not None:
            stop_at_goal = np.cos(stop_at_goal*TORAD / 2)

        N = self.N

        time_start = time.time()
        time_end = time.time()
        print(f'progress: {0}%, {0}/{self.N}, elapsed: '
              f'{time_end-time_start:.2f}')

        for i in range(self.i+1, self.N):

            # calculate all forces
            self.control(i)  # attitude control
            self.motor_model(i)  # get motor commands for given control
            self.gravity_and_aero(i)

            # integration and reconstruction
            etad = self.get_accelerations(i)
            self.integrate(etad, i)

            # noisy measurements unreliable measurements
            self.get_measured_data(i)
            self.estimate_drift_angle(i)

            if stop_at_goal is not None and self.nmiddle[i].dot(self.nmiddledot[i]) >= stop_at_goal:
                break

            if self.gamma_d != 0 and norm(self.w[i]) > 10*self.gamma_d:
                break

            if i % self.dN == 0:
                time_end = time.time()
                elapsed = time_end-time_start
                elapsed_min = int(elapsed / 60)
                elapsed_sec = elapsed - 60*elapsed_min
                remaining = elapsed * (N/i - 1)
                remaining_min = int(remaining / 60)
                remaining_sec = remaining - 60*remaining_min
                print(f'progress: {100*i/N:.0f}%, '
                      f'{i}/{N}, elapsed: {elapsed_min}:{elapsed_sec:.0f}, '
                      f'remaining: {remaining_min}:{remaining_sec:.0f}, ')

        print('****************************************')
        print('* End of test at i = {}/{}'.format(i, N-1))
        print('* Time t = {}/{}'.format(self.t[i], self.t[-1]))
        print('****************************************')
        self.i = i

        if i < self.N-1:
            self.trim_results(i)

    def control(self, i):
        q_ = self.q_measured[i-1]
        nmiddle_ = self.nmiddle_measured[i-1]
        w_ = self.w_measured[i-1]
        nmiddledot_ = self.nmiddledot[i-1]
        n = self.ctrl_order

        C = np.cross(nmiddle_, nmiddledot_)
        D1 = np.dot(nmiddle_, nmiddledot_)
        D2 = np.dot(nmiddle_, ez)

        D1 = D1 if abs(D1) > eps else 0
        D2 = D2 if abs(D2) > eps else 0

        S = np.sign(D1) ** (n - 1)

        w_spin = (nmiddle_.dot(w_)/D2) * ez
        w_ctrl = q_.inverse.rotate(C/(D1**n))
        # self.w_delta[i] = (w_.dot(w_))*ez + self.P @ q_.inverse.rotate(S * C/(D1**n))
        self.w_delta[i] = w_spin + S * self.P @ w_ctrl

        # angular rate control
        self.w_delta[i] = self.w_delta[i] - self.w_measured[i-1]
        self.w_delta_der[i] = (self.w_delta[i] - self.w_delta[i-1])/self.DT
        self.w_delta_int[i] = self.w_delta_int[i-1] + self.w_delta[i]*self.DT

#        self.t_ctrl[i] = self.Kp * self.w_delta[i-1] + self.Kd * self.w_delta_der[i-1] + self.Ki * self.w_delta_int[i-1]
        self.t_ctrl[i] = self.Kp * self.w_delta[i] + self.Kd * self.w_delta_der[i] + self.Ki * self.w_delta_int[i]

        self.f_ctrl[i] = -np.cross(ez, self.t_ctrl[i])/self.hp

        # low pass filter
        if self.ctrl_window > 1:
            self.f_ctrl[i] = np.mean(
                    self.f_ctrl[max(0, i+1-self.ctrl_window):i+1], axis=0)
            self.t_ctrl[i] = np.mean(
                    self.t_ctrl[max(0, i+1-self.ctrl_window):i+1], axis=0)

    def motor_model(self, i):
        ftot = self.f_ctrl[i] + self.fz

        # propeller force and torque
        self.rotvel[i] = np.sqrt(norm(ftot)/self.kf)
        self.rotvel[i] = max(
                self.rotvel[i], self.rotvel_diff_lim[0]*self.gamma_d)
        self.rotvel[i] = min(
                self.rotvel[i], self.rotvel_diff_lim[1]*self.gamma_d)

        # get rotor angles
        ftot = ftot/norm(ftot)
        self.angles[i][1] = np.arccos(ftot[2])
        self.angles[i][0] = np.arctan2(ftot[1], ftot[0])

        if np.isnan(self.angles[i][1]):
            self.angles[i][1] = 0
        if np.isnan(self.angles[i][0]):
            self.angles[i][0] = 0

        # limit rotor angles
        if self.angles[i][1] < self.alpha_lim[0]:
            self.angles[i][1] = 0
        elif self.angles[i][1] > self.alpha_lim[1]:
            self.angles[i][1] = self.alpha_lim[1]

#        if self.angles[i][1] < 10E-1:
#            self.angles[i][0] = 0

        Omega = 0*self.w[i-1][-1] + self.rotvel[i]
        fp = self.kf * Omega * Omega
        r = quat.array.from_euler_angles(*self.angles[i-1]).rotate(ez)
        self.f_prop[i] = fp * r
        self.t_prop[i] = self.B @ self.f_prop[i]

    def gravity_and_aero(self, i):
        g_B = self.q[i-1].rotate(-self.g * ez)
        self.t_grav[i] = -self.md * np.cross(ez, g_B)
        self.f_grav[i] = +self.mT * g_B

        # aerodynamic drag torque
        self.t_aero[i] = -norm(self.w[i-1]) * self.Kaero @ self.w[i-1]
#        self.t_aero[i] = -norm(self.w) * self.Kaero @ self.w[i-1]

    def get_accelerations(self, i, gymbal_fixed=False):
        w_ = self.w[i-1]
        v_ = self.v[i-1]

        # all external forces and torques
        Fext = np.append(
                self.t_prop[i] + self.t_grav[i] + self.t_aero[i] + self.t_rndn[i] + self.t_imp[i],
                self.f_prop[i] + self.f_grav[i] + self.f_rndn[i] + self.f_imp[i])

        # coupling torque term
        RBP = quat.array.from_euler_angles(self.angles[i-1]).to_rotation_matrix
        t_coup = self.Jp[2, 2] * np.cross(self.w[i-1], RBP @ ez)

        # full transformed inertia matrix
        JT = self.Jb + RBP @ self.Jp @ (RBP.T)

        # EOM components
        D = mstack([[JT, zeroM], [zeroM, self.mI]]) + self.mx
        Dinv = np.linalg.inv(D)
        C = -ad(w_, v_).T @ D

        eta = np.append(w_, v_)
        etad = Dinv @ (Fext - np.hstack([t_coup, zeroV]) - C @ eta)

        if gymbal_fixed:
            etad[3:] = [0, 0, 0]

        return etad

    def integrate(self, etad, i):
        # TODO: replace these integration steps for a better algorithm
        # Example: RK for angular acceleration, and SLERP for quaternion

        DT = self.DT
        self.w[i] = self.w[i-1] + DT*etad[:3]
        self.v[i] = self.v[i-1] + DT*etad[3:]
        self.angles[i][2] = self.angles[i-1][2] + self.rotvel[i-1]*DT

        qd = DT * 0.5 * self.q[i-1] * quat.array.from_vector_part(self.w[i-1])
        self.q[i] = (self.q[i-1] + qd).normalized
        self.pos[i] = self.pos[i-1] + DT * self.q[i-1].rotate(self.v[i-1])

        self.nmiddle[i] = get_middle_vector(self.q[i], self.nmiddle[i-1])

        self.pre[i],self.nut[i],self.spin[i] = self.q[i].to_euler_angles.T
        self.spin[i] += self.pre[i]

    # for future drift-related tests:
    def get_measured_data(self, i):
        self.drift_angle[i] = self.drift_angle[i-1] + self.DT*self.drift_rate
        self.q_drift[i] = quat.array.from_axis_angle(self.drift_angle[i] * ez)
        self.w_measured[i] = self.w[i] + (2*np.random.rand(3)-1)*self.noise_measures[0]
        self.q_measured[i] = self.q[i] + quat.array.random(normalize=1)*self.noise_measures[1]
        self.q_measured[i] = (self.q_measured[i]*self.q_drift[i]).normalized
        self.nmiddle_measured[i] = get_middle_vector(self.q_measured[i], self.nmiddle_measured[i])

    def estimate_drift_angle(self, i):

        def get_drift_angle(q, w, nmiddle):
            Rw = q.rotate(w)
            wb = ez * nmiddle.dot(Rw) / nmiddle.dot(ez)
            wnmiddle = q.rotate(w - wb)
            nmiddledot = -0.5 * np.cross(nmiddle, wnmiddle)
            b = nmiddle.dot(ez) * nmiddledot.dot(ez)
            d = np.cross(nmiddle, nmiddledot).dot(ez)
            return np.arctan2(d, b)

        self.drift_angle_measured[i] = get_drift_angle(
                self.q_measured[i],
                self.nmiddle_measured[i],
                self.w_measured[i])

    def trim_results(self, i=None):
        if i is None:
            i = self.i
        for att in arrlist_1 + arrlist_3 + arrlist_quat:
            setattr(self, att, getattr(self, att)[:i])

    # parameter setting helpers
    def set_parameters(self, parameters, init_arrays=True):
        self.parameters = parameters

        self.g = parameters['g']
        hs = self.hs = parameters['hs']
        hp = self.hp = parameters['hp']

        mb = self.mb = parameters['mb']
        ms = self.ms = parameters['ms']
        mp = self.mp = parameters['mp']
        mT = self.mT = mb+ms+mp
        md = self.md = hs*ms - hp*mp
        self.mI = mT * np.eye(3)

        self.mx = md * mstack([[zeroM, -Ez],
                              [Ez, zeroM]])

        self.r = np.array([0, 0, hp])
        self.r.shape = (3, 1)
        self.mpr = mp*self.r
        self.mprx = cross(self.mpr.T[0])

        Jpx = self.Jpx = parameters['Jpx']
        Jpy = self.Jpy = parameters['Jpy']
        Jpz = self.Jpz = parameters['Jpz']
        Jpxy = self.Jpxy = (Jpx+Jpy)/2
        self.Jp = np.diag([Jpxy, Jpxy, Jpz])

        Jbx = self.Jbx = parameters['Jbx']
        Jby = self.Jby = parameters['Jby']
        Jbz = self.Jbz = parameters['Jbz']
        self.Jb = np.diag([Jbx, Jby, Jbz])

        Jsx = self.Jsx = parameters['Jsx']
        Jsy = self.Jsy = parameters['Jsy']
        Jsz = self.Jsz = parameters['Jsz']
        self.Js = np.diag([Jsx, Jsy, Jsz])

        self.Jb = self.Jb + self.Js + np.diag([1, 1, 0])*(ms*(hs**2) + mp*(hp**2))

        self.ktau = parameters['ktau']
        self.kf = parameters['kf']
        self.kratio = self.ktau / self.kf

        self.B = self.kratio*Id + hp*Ez
        self.Binv = np.linalg.inv(self.B)
        self.BTotal = self.Binv @ self.Jb / self.kf

        self.angvelz_ratio = parameters['angvelz_ratio']
        self.angvelxy_coef = parameters['angvelxy_coef']
#        self.angvelz_ratio = parameters['angvelz_ratio']

        # Drag
        if self.angvelz_ratio == 0:
            self.Kz = 1000000
        else:
            self.Kz = self.ktau * (1/self.angvelz_ratio + 1)**2

        self.Kaero = np.diag([self.angvelxy_coef, self.angvelxy_coef, 1]) * self.Kz

        # Forces
        self.FZ = self.mT * self.g
        self.fz = self.FZ * ez
        self.tz = self.B @ self.fz
        self.gamma_d = 1.0 * np.sqrt(self.FZ / self.kf)  # rotor mean velocity

        self.Nmin = Nmin = parameters['Nmin']
        self.Nmin = Nmax = parameters['Nmax']
        self.DT = DT = parameters['DT']
        self.tmax = tmax = parameters['tmax']

        self.t = np.arange(0, tmax+DT, DT)
        self.N = len(self.t)

        if Nmin is not None and self.N < Nmin:
            self.DT = tmax/Nmin
            self.t = np.array(range(Nmin))*DT
            self.N = Nmin
        elif Nmax is not None and self.N > Nmax:
            self.DT = tmax/Nmax
            self.t = np.array(range(Nmax))*DT
            self.N = Nmax
        self.dN = int(self.N / (100/parameters['progress_warning_percentage']))

        # setting zero arrays
        self.q = quat.array(np.zeros([self.N, 4]))
        self.q_measured = quat.array(np.zeros([self.N, 4]))
        self.q_drift = quat.array(np.zeros([self.N, 4]))
        self.nmiddle = np.zeros([self.N, 3])
        self.nmiddle_measured = np.zeros([self.N, 3])
        self.w = np.zeros([self.N, 3])
        self.w_measured = np.zeros([self.N, 3])
        self.w_delta = np.zeros([self.N, 3])
        self.w_delta_int = np.zeros([self.N, 3])
        self.w_delta_der = np.zeros([self.N, 3])
        self.t_ctrl = np.zeros([self.N, 3])
        self.f_ctrl = np.zeros([self.N, 3])
        self.t_grav = np.zeros([self.N, 3])
        self.f_grav = np.zeros([self.N, 3])
        self.t_prop = np.zeros([self.N, 3])
        self.f_prop = np.zeros([self.N, 3])
        self.t_aero = np.zeros([self.N, 3])
        self.t_rndn = np.zeros([self.N, 3])
        self.f_rndn = np.zeros([self.N, 3])
        self.t_imp = np.zeros([self.N, 3])
        self.f_imp = np.zeros([self.N, 3])
#        self.Fext = np.zeros(6)
        self.alpha = np.zeros(self.N)
        self.beta = np.zeros(self.N)
        self.Xd = np.zeros([self.N, 3])
        self.drift_angle = np.zeros(self.N)
        self.drift_angle_measured = np.zeros(self.N)
        self.angles = np.zeros([self.N, 3])
        self.rotvel = np.zeros(self.N)
        self.pos = np.zeros([self.N, 3])
        self.v = np.zeros([self.N, 3])
        self.pre = np.zeros(self.N)
        self.nut = np.zeros(self.N)
        self.spin = np.zeros(self.N)

        # set control parameters
        self.ctrl_window = parameters['window']
        self.alpha_lim = np.array(parameters['alpha_lim_deg'])*TORAD
        self.rotvel_diff_lim = parameters['rotvel_diff_lim']

        # control parameters
        self.P = np.diag(parameters['P']) * self.mT * self.g
        self.Kp = parameters['Kp']
        self.Kd = parameters['Kd']
        self.Ki = parameters['Ki']
        self.ctrl_order = parameters['order']

        # set noise
#        self.drift_is_constant = self.parameters['drift_is_constant']
        self.drift_rate = parameters['drift_rate_rads']
        self.drift_flip = parameters['drift_flip']
        self.drift_angle_init = parameters['drift_angle_init_deg']*TORAD

        self.noise_measures = self.parameters['noise_measures']
        self.noise_random = self.parameters['noise_random']
        self.noise_impulses = self.parameters['noise_impulses']

        self.number_impulses = self.parameters['number_impulses']

        # initual condition
        zyz = np.array(self.parameters['init_orientation_zyz_deg'])*TORAD
        self.q0 = quat.array.from_euler_angles(*zyz)
        self.pos0 = self.parameters['init_position']
        self.init_spin_percentage = 0.3

        # goal orientation
        alphad = self.parameters['alpha_deg']
        betad = self.parameters['beta_deg']

        if type(alphad) != list:
            alphad = [alphad]
        if type(betad) != list:
            betad = [betad]

        ngoal = len(alphad)
        if ngoal != len(betad):
            raise ValueError('alphad and betad must have the same length')

        alphad = np.array(alphad)*TORAD
        betad = np.array(betad)*TORAD
        self.ngoal = ngoal

        sa, ca = np.sin(alphad), np.cos(alphad)
        sb, cb = np.sin(betad), np.cos(betad)

        self.nd_ = np.array([sa*cb, sa*sb, ca]).T

        if init_arrays:
            self.init_arrays()
