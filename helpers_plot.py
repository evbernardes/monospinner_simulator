#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 08:59:32 2021

@author: evbernardes
"""
import matplotlib.pyplot as plt
import numpy as np

from helpers import angdiff, wrap, ez, TODEG, TORAD, quat_to_XYZ
import quaternionic as quat
#
#from helpers_indices import iQA, iQX, iQY, iQZ, iX, iY, iZ, x_len, iQ_, iX_
#from helpers_indices import iWX, iWY, iWZ, iVX, iVY, iVZ, eta_len, iW_, iV_
#from helpers_indices import iPX, iPY, iPZ, error_len, iP_
#from helpers_indices import iBETA, iALPHA, iGAMMA, iBETA_D, iALPHA_D, iGAMMA_D, iBETA_DD, iALPHA_DD, iGAMMA_DD, iBETA_D0, iALPHA_D0, iGAMMA_D0, angles_len
#from helpers_indices import x0, angles0, eta0, error0

def subplot(a,*vargin,**kargs): 
    if len(vargin) == 0:
        if type(a) != int or a < 1000 or a > 9999:
            raise ValueError('Single argument to subplot must be a 4-digit integer')
        rows = int(a/1000)
        cols = int((a-rows*1000)/100)
        r = int((a-rows*1000-cols*100)/10)
        c = int(a-rows*1000-cols*100-r*10)
    elif len(vargin) == 3:
        for v in vargin:
            if type(v) != int:
                raise ValueError('Illegal argument(s) to subplot: {}'.format((a,)+vargin))
        rows = a
        cols,r,c = vargin
    else:
        raise ValueError('Illegal argument(s) to subplot: {}'.format((a,)+vargin))
        
    return plt.subplot(rows,cols,c + cols*(r-1),**kargs)  

#####################################
# PLOT METHODS
#####################################
def plot_vectors(t, n, v, rotor_angles, vd, tight = True, title = 'vectors', alphamax = 10*TORAD):
    plt.figure(title)
    plt.clf()
    
    subplot(3311); plt.plot(t, rotor_angles.T[0]*TODEG,'m'); plt.ylabel(r'$\beta$'); plt.ylim([-190, 190]) #plt.ylim([-0.1, 15*np.pi/180]); plt.title('ang')
    subplot(3321); plt.plot(t, rotor_angles.T[1]*TODEG,'c'); plt.ylabel(r'$\alpha$'); plt.ylim([-(alphamax*TODEG*1.2), (alphamax*TODEG*1.2)]); #plt.ylim([-np.pi, np.pi])
    subplot(3331); plt.plot(t, rotor_angles.T[2],'k'); plt.ylabel(r'$\dot{\gamma}$'); #plt.ylim([-1.1, 1.1]); #plt.ylim([-np.pi, np.pi])
    
    subplot(3312); plt.plot(t, n.T[0],'r'); plt.ylabel(r'$x$'); plt.title('n'); plt.ylim([-1.1, 1.1]) #plt.ylim([-0.1, 15*np.pi/180]); plt.title('ang')
    subplot(3322); plt.plot(t, n.T[1],'g'); plt.ylabel(r'$y$'); plt.ylim([-1.1, 1.1]); #plt.ylim([-np.pi, np.pi])
    subplot(3332); plt.plot(t, n.T[2],'b'); plt.ylabel(r'$z$'); plt.ylim([-1.1, 1.1]); #plt.ylim([-np.pi, np.pi])
#    plt.title('vel')
    subplot(3313); plt.plot(t, v.T[0],'r'); plt.ylim([-1.1, 1.1]); plt.title('v')
    plt.hlines(vd[0], t[0], t[-1], colors='r', linestyles='dashed')
    subplot(3323); plt.plot(t, v.T[1],'g'); plt.ylim([-1.1, 1.1]); #plt.lim([-np.pi, np.pi])
    plt.hlines(vd[1], t[0], t[-1], colors='g', linestyles='dashed')
    subplot(3333); plt.plot(t, v.T[2],'b'); plt.ylim([-1.1, 1.1]); #plt.lim([-np.pi, np.pi])
    plt.hlines(vd[2], t[0], t[-1], colors='b', linestyles='dashed')
#    plt.title('acc')

def plot_angles(t, angles, tight=True, title='rotor angles'):
    plt.figure(title)
    plt.clf()
#    plt.title('ang')
    subplot(3111); plt.plot(t, angles.T[0],'r'); plt.ylabel(r'$\beta$'); #plt.ylim([-0.1, 15*np.pi/180]); plt.title('ang')
    subplot(3121); plt.plot(t, angles.T[1],'g'); plt.ylabel(r'$\alpha$'); #plt.ylim([-np.pi, np.pi])
    subplot(3131); plt.plot(t, angles.T[2],'b'); plt.ylabel(r'$\gamma$'); #plt.ylim([-np.pi, np.pi])
   
    if tight:
        plt.tight_layout()
        
def plot_eta(t, eta, tight=True, title='body velocities'):
    plt.figure(title)
    plt.clf()
    subplot(3211); plt.plot(t, eta.T[iVX],'r'); plt.title(r'$v_B$')
    subplot(3221); plt.plot(t, eta.T[iVY],'g'); 
    subplot(3231); plt.plot(t, eta.T[iVZ],'b'); 
    
    subplot(3212); plt.plot(t, eta.T[iWX],'r'); plt.title(r'$\omega_B$')
    subplot(3222); plt.plot(t, eta.T[iWY],'g'); 
    subplot(3232); plt.plot(t, eta.T[iWZ],'b'); 
   
    if tight:
        plt.tight_layout()
        
def plot_x(t, x, tight=True, title='body pos'):
    plt.figure(title)
#    qr,qx,qy,qz = (x.T[iQA:iQZ+1])
    qr,qx,qy,qz = (x.T[iQ_])
    
    qb_norm = np.sqrt(qr * qr + qz * qz)
    qb = np.zeros([4,len(qr)])
    qb[0] = qr/qb_norm
    qb[3] = qz/qb_norm
#    qb = qb / qb_norm
    
    qa = np.zeros(qb.shape)
    qa[0] = qb_norm
    qa[1] = (qr*qx - qy*qz)/qb_norm
    qa[2] = (qr*qy + qx*qz)/qb_norm
    qa = qa / np.linalg.norm(qa)
    
    
    plt.clf()
#    plt.title('ang')
    subplot(4411); plt.plot(t, x.T[iX],'r'); plt.title(r'$x_E$')
    subplot(4421); plt.plot(t, x.T[iY],'g'); 
    subplot(4431); plt.plot(t, x.T[iZ],'b'); 
    
#    plt.title('vel')
    subplot(4442); plt.plot(t, qr,'k'); plt.ylim([-1.1, 1.1])
    subplot(4412); plt.plot(t, qx,'r'); plt.ylim([-1.1, 1.1]); plt.title(r'$q_{EB}$')
    subplot(4422); plt.plot(t, qy,'g'); plt.ylim([-1.1, 1.1])
    subplot(4432); plt.plot(t, qz,'b'); plt.ylim([-1.1, 1.1])
    
    subplot(4443); plt.plot(t, qa[0],'k'); plt.ylim([-1.1, 1.1])
    subplot(4413); plt.plot(t, qa[1],'r'); plt.ylim([-1.1, 1.1]); plt.title(r'$q_{xy}$')
    subplot(4423); plt.plot(t, qa[2],'g'); plt.ylim([-1.1, 1.1])#plt.lim([-np.pi, np.pi])
    subplot(4433); plt.plot(t, qa[3],'b'); plt.ylim([-1.1, 1.1])#plt.lim([-np.pi, np.pi])
    
    subplot(4444); plt.plot(t, qb[0],'k'); plt.ylim([-1.1, 1.1])
    subplot(4414); plt.plot(t, qb[1],'r'); plt.ylim([-1.1, 1.1]); plt.title(r'$q_{z}$')
    subplot(4424); plt.plot(t, qb[2],'g'); plt.ylim([-1.1, 1.1])#plt.lim([-np.pi, np.pi])
    subplot(4434); plt.plot(t, qb[3],'b'); plt.ylim([-1.1, 1.1])#plt.lim([-np.pi, np.pi])
#    plt.title('acc')
    if tight:
        plt.tight_layout()
        
def plot_q_omega(t, q, w, e = np.array([0, 0, 1]), tight=True, title='q and omega'):
    plt.figure(title)
    
    e_ = np.hstack([1, e])[np.newaxis]
    qb = q.copy()
    qb.vector = ((e[np.newaxis].T * e) @ qb.vector.T).T
    qb = qb.normalized
    
    Qhat = Quat.Qa(e)

    qr, qx, qy, qz = q.T
#    qb_norm = np.sqrt(qr * qr + qz * qz)
#    qb = np.zeros([4,len(qr)])
#    qb[0] = qr/qb_norm
#    qb[3] = qz/qb_norm
    qa = np.zeros(qb.shape)
    qa[0] = qb_norm
    qa[1] = (qr*qx - qy*qz)/qb_norm
    qa[2] = (qr*qy + qx*qz)/qb_norm
    
    plt.clf()
#    plt.title('ang')
    
    subplot(4411); plt.plot(t, w.T[0],'r'); plt.title(r'$\omega_B$')
    subplot(4421); plt.plot(t, w.T[1],'g'); 
    subplot(4431); plt.plot(t, w.T[2],'b'); 
   
    subplot(4442); plt.plot(t, qr,'k'); plt.ylim([-1.1, 1.1])
    subplot(4412); plt.plot(t, qx,'r'); plt.ylim([-1.1, 1.1]); plt.title(r'$q = q_{xy} \times q_{z}$')
    subplot(4422); plt.plot(t, qy,'g'); plt.ylim([-1.1, 1.1])
    subplot(4432); plt.plot(t, qz,'b'); plt.ylim([-1.1, 1.1])
    
    subplot(4443); plt.plot(t, qa[0],'k'); plt.ylim([-1.1, 1.1])
    plt.hlines(Qhat[0], t[0], t[-1], colors='k', linestyles='dashed')
    subplot(4413); plt.plot(t, qa[1],'r'); plt.ylim([-1.1, 1.1]); plt.title(r'$q_{xy}$')
    plt.hlines(Qhat[1], t[0], t[-1], colors='r', linestyles='dashed')
    subplot(4423); plt.plot(t, qa[2],'g'); plt.ylim([-1.1, 1.1])#plt.lim([-np.pi, np.pi])
    plt.hlines(Qhat[2], t[0], t[-1], colors='g', linestyles='dashed')
    subplot(4433); plt.plot(t, qa[3],'b'); plt.ylim([-1.1, 1.1])#plt.lim([-np.pi, np.pi])
    plt.hlines(Qhat[3], t[0], t[-1], colors='b', linestyles='dashed')
    
    subplot(4444); plt.plot(t, qb[0],'k'); plt.ylim([-1.1, 1.1])
    subplot(4414); plt.plot(t, qb[1],'r'); plt.ylim([-1.1, 1.1]); plt.title(r'$q_{z}$')
    subplot(4424); plt.plot(t, qb[2],'g'); plt.ylim([-1.1, 1.1])#plt.lim([-np.pi, np.pi])
    subplot(4434); plt.plot(t, qb[3],'b'); plt.ylim([-1.1, 1.1])#plt.lim([-np.pi, np.pi])
    
#    roll = []
#    pitch = []
#    yaw = []
#    for i in range(len(t)):   
#        roll_,pitch_,yaw_ = quat_to_XYZ([qr[i],qx[i],qy[i],qz[i]])
#        roll.append(roll_*TODEG)
#        pitch.append(pitch_*TODEG)
#        yaw.append(yaw_*TODEG)
#        
#    subplot(4641); plt.title('roll'); plt.plot(t, roll,'r');
#    subplot(4642); plt.title('pitch'); plt.plot(t, pitch,'g');
#    subplot(4643); plt.title('yaw'); plt.plot(t, yaw,'b');
#    plt.title('acc')
    if tight:
        plt.tight_layout()
        
def plot_x_eta(sim, i = None, tight = True):
    
#    t,q,q_measured,w,w_measured,pos,v,k,k_measured, kd, wz_, angles, rotvel = sim.get_values(i)
    
#    Qhat = Quat.Qa(e)
#    qr,qx,qy,qz = (x.T[iQ_])
    t = sim.t
    qr = sim.q.real
    qx,qy,qz = sim.q.vector.T
    qb_norm = np.sqrt(qr * qr + qz * qz)
    qb = np.zeros([4,len(qr)])
    qb[0] = qr/qb_norm
    qb[3] = qz/qb_norm
    qa = np.zeros(qb.shape)
    qa[0] = qb_norm
    qa[1] = (qr*qx - qy*qz)/qb_norm
    qa[2] = (qr*qy + qx*qz)/qb_norm
    
    if sim.q_measured is not None:
        qmr = sim.q_measured.real
        qmx,qmy,qmz = sim.q_measured.vector.T
        qmb_norm = np.sqrt(qmr * qmr + qmz * qmz)
        qmb = np.zeros([4,len(qmr)])
        qmb[0] = qmr/qmb_norm
        qmb[3] = qmz/qmb_norm
        qma = np.zeros(qmb.shape)
        qma[0] = qmb_norm
        qma[1] = (qmr*qmx - qmy*qmz)/qmb_norm
        qma[2] = (qmr*qmy + qmx*qmz)/qmb_norm
    
    plt.clf()
#    plt.title('ang')
    subplot(4612); plt.plot(t, sim.pos.T[0],'r'); plt.title(r'$x^E$')
    subplot(4622); plt.plot(t, sim.pos.T[1],'g'); 
    subplot(4632); plt.plot(t, sim.pos.T[2],'b'); 
    
    subplot(4611); plt.plot(t, sim.v.T[0],'r'); plt.title(r'$v^B$')
    subplot(4621); plt.plot(t, sim.v.T[1],'g'); 
    subplot(4631); plt.plot(t, sim.v.T[2],'b'); 
    
    if sim.w_measured is not None:
        subplot(4613); plt.plot(t, sim.w_measured.T[0],'m');
        subplot(4623); plt.plot(t, sim.w_measured.T[1],'y'); 
        subplot(4633); plt.plot(t, sim.w_measured.T[2],'c');
        
    subplot(4613); plt.plot(t, sim.w.T[0],'r'); plt.title(r'$omega_B$')
    subplot(4623); plt.plot(t, sim.w.T[1],'g'); 
    subplot(4633); plt.plot(t, sim.w.T[2],'b'); 
    
    # terminal angular velocity
    if sim.terminal_wz is not None:
        plt.hlines(sim.terminal_wz, t[0], t[-1], 'k', linestyles='dashed')
   
    
    if sim.q_measured is not None:
        subplot(4644); plt.plot(t, qmr,'gray'); plt.ylim([-1.1, 1.1])
        subplot(4614); plt.plot(t, qmx,'m'); plt.ylim([-1.1, 1.1])#; plt.title(r'$q = q_{xy} \times q_{z}$')
        subplot(4624); plt.plot(t, qmy,'y'); plt.ylim([-1.1, 1.1])
        subplot(4634); plt.plot(t, qmz,'c'); plt.ylim([-1.1, 1.1])
        
        subplot(4645); plt.plot(t, qma[0],'gray'); plt.ylim([-1.1, 1.1])
        subplot(4615); plt.plot(t, qma[1],'m'); plt.ylim([-1.1, 1.1])#; plt.title(r'$q_{xy}$')
        subplot(4625); plt.plot(t, qma[2],'y'); plt.ylim([-1.1, 1.1])
        subplot(4635); plt.plot(t, qma[3],'c'); plt.ylim([-1.1, 1.1])
        
        subplot(4646); plt.plot(t, qmb[0],'gray'); plt.ylim([-1.1, 1.1])
        subplot(4616); plt.plot(t, qmb[1],'m'); plt.ylim([-1.1, 1.1])#; plt.title(r'$q_{z}$')
        subplot(4626); plt.plot(t, qmb[2],'y'); plt.ylim([-1.1, 1.1])#plt.lim([-np.pi, np.pi])
        subplot(4636); plt.plot(t, qmb[3],'c'); plt.ylim([-1.1, 1.1])#plt.lim([-np.pi, np.pi])
    
        
    subplot(4644); plt.plot(t, qr,'k'); plt.ylim([-1.1, 1.1])
    subplot(4614); plt.plot(t, qx,'r'); plt.ylim([-1.1, 1.1]); plt.title(r'$q = q_{xy} \times q_{z}$')
    subplot(4624); plt.plot(t, qy,'g'); plt.ylim([-1.1, 1.1])
    subplot(4634); plt.plot(t, qz,'b'); plt.ylim([-1.1, 1.1])
    
    subplot(4645); plt.plot(t, qa[0],'k'); plt.ylim([-1.1, 1.1])
    subplot(4615); plt.plot(t, qa[1],'r'); plt.ylim([-1.1, 1.1]); plt.title(r'$q_{xy}$')
    subplot(4625); plt.plot(t, qa[2],'g'); plt.ylim([-1.1, 1.1])
    subplot(4635); plt.plot(t, qa[3],'b'); plt.ylim([-1.1, 1.1])
    
    subplot(4646); plt.plot(t, qb[0],'k'); plt.ylim([-1.1, 1.1])
    subplot(4616); plt.plot(t, qb[1],'r'); plt.ylim([-1.1, 1.1]); plt.title(r'$q_{z}$')
    subplot(4626); plt.plot(t, qb[2],'g'); plt.ylim([-1.1, 1.1])#plt.lim([-np.pi, np.pi])
    subplot(4636); plt.plot(t, qb[3],'b'); plt.ylim([-1.1, 1.1])#plt.lim([-np.pi, np.pi])
    
    roll = []
    pitch = []
    yaw = []
    for i in range(len(t)):   
        roll_,pitch_,yaw_ = quat_to_XYZ([qr[i],qx[i],qy[i],qz[i]])
        roll.append(roll_*TODEG)
        pitch.append(pitch_*TODEG)
        yaw.append(yaw_*TODEG)
        
    subplot(4641); plt.title('roll'); plt.plot(t, roll,'r');
    subplot(4642); plt.title('pitch'); plt.plot(t, pitch,'g');
    subplot(4643); plt.title('yaw'); plt.plot(t, yaw,'b');
#    plt.title('acc')
    if tight:
        plt.tight_layout()
        
#def plot_k_omega(t, q, w, angles, rotvel, nd = np.array([0, 0, 1]), tight=True, wz_ = None, w_measured = None, q_measured = None):
def plot_k_omega(sim, i = None, tight = True):
    
#    t,q,q_measured,w,w_measured,pos,v,k,k_measured, kd, wz_, angles, rotvel = sim.get_values(i)
    t = sim.t
    
    m = 3 + (sim.q_measured is not None)
    
    plt.clf()
    
    if sim.w_measured is not None:
        subplot(m,4,1,1); plt.plot(t, sim.w_measured.T[0],'m');
        subplot(m,4,2,1); plt.plot(t, sim.w_measured.T[1],'y'); 
        subplot(m,4,3,1); plt.plot(t, sim.w_measured.T[2],'c'); 
    
    subplot(m,4,1,1); plt.plot(t, sim.w.T[0],'r'); plt.title(r'$\omega_B$')
    subplot(m,4,2,1); plt.plot(t, sim.w.T[1],'g'); 
    subplot(m,4,3,1); plt.plot(t, sim.w.T[2],'b'); 
    
    # terminal angular velocity
    if sim.terminal_wz is not None:
        plt.hlines(sim.terminal_wz, t[0], t[-1], 'k', linestyles='dashed')
    
    pre, nut, spin = wrap(sim.q.to_euler_angles).T
    spin = wrap(spin + pre)
    
    qd = -quat.array.from_vector_part(sim.kd) * quat.array.from_vector_part(ez)
    pred, nutd, spind = wrap(qd.to_euler_angles).T
    
    if sim.q_measured is not None:
#        pre_measured, nut_measured, spin_measured = wrap(q_measured.to_euler_angles).T
#        spin_measured = wrap(spin_measured + pre_measured)
#        
#        subplot(3412); plt.plot(t, pre_measured*TODEG,'r'); plt.title(r'$precession$'); plt.ylim([-190, 190]);
#        subplot(3422); plt.plot(t, nut_measured*TODEG,'g'); plt.title(r'$nutation$'); plt.ylim([-10, 190]);
        spin_measured = wrap(2*np.arctan2(sim.q_measured.z, sim.q_measured.real))
        subplot(m,4,3,2); plt.plot(t, spin_measured*TODEG,'c')#; plt.title(r'$spin$'); plt.ylim([-190, 190]);
        subplot(m,4,4,2); plt.plot(t, angdiff(spin,spin_measured)*TODEG,'c'); plt.title(r'$\Delta$ spin'); plt.ylim([-190, 190]);
    
    subplot(m,4,1,2); plt.plot(t, pred*TODEG,'k'); plt.plot(t, pre*TODEG,'r'); plt.title(r'$precession$'); plt.ylim([-190, 190]);
    subplot(m,4,2,2); plt.plot(t, nutd*TODEG,'k'); plt.plot(t, nut*TODEG,'g'); plt.title(r'$nutation$'); plt.ylim([-10, 190]);
    subplot(m,4,3,2); plt.plot(t, spin*TODEG,'b'); plt.title(r'$spin$'); plt.ylim([-190, 190]);
#    subplot(4342); plt.plot(t, (spin - 2*pre)*TODEG,'b'); plt.title(r'$s$')
    
    
    
    subplot(m,4,1,3); 
    plt.plot(t, sim.kd[:, 0], 'k')
    plt.plot(t, sim.k[:, 0],'r'); plt.ylim([-1.1, 1.1]);
    plt.title(r'middle vector $(k)$')
#    plt.hlines(kd[0], t[0], t[-1], colors='k', linestyles='dashed')
    subplot(m,4,2,3); 
    plt.plot(t, sim.kd[:, 1], 'k')
    plt.plot(t, sim.k[:, 1],'g'); plt.ylim([-1.1, 1.1])
#    plt.hlines(kd[1], t[0], t[-1], colors='k', linestyles='dashed')
    subplot(m,4,3,3); 
    plt.plot(t, sim.kd[:, 2], 'k')
    plt.plot(t, sim.k[:, 2],'b'); plt.ylim([-0.1, 1.1])
#    plt.hlines(kd[2], t[0], t[-1], colors='k', linestyles='dashed')
   
#    subplot(4343); plt.plot(t, phi,'k'); #plt.ylim([-1.1, 1.1])
    
    
    #%% rotor variables
    subplot(m,4,1,4); plt.plot(t, sim.angles.T[0]*TODEG,'r'); plt.ylabel(r'$\beta$'); #plt.ylim([-0.1, 15*np.pi/180]); plt.title('ang')
    subplot(m,4,2,4); plt.plot(t, sim.angles.T[1]*TODEG,'g'); plt.ylabel(r'$\alpha$'); #plt.ylim([-np.pi, np.pi])
    subplot(m,4,3,4); plt.plot(t, sim.rotvel,'b'); plt.ylabel(r'$\dot \gamma$ rad/s'); #plt.ylim([-np.pi, np.pi])
#    print(list(plt.yticks()[0]))
#    plt.yticks(list(plt.yticks()[0]) + np.median(rotvel));# plt.ylim([min(rotvel)-200, max(rotvel)+200])
    
    if tight:
        plt.tight_layout()


def plot_error(t, x, eta, tight=True, title='error'):
    plt.figure(title)

    qr,qx,qy,qz = (x.T[iQ_])
    qb_norm = np.sqrt(qr * qr + qz * qz)
    qb = np.zeros([4,len(qr)])
    qb[0] = qr/qb_norm
    qb[3] = qz/qb_norm
    qa = np.zeros(qb.shape)
    qa[0] = qb_norm
    qa[1] = (qr*qx - qy*qz)/qb_norm
    qa[2] = (qr*qy + qx*qz)/qb_norm
    
    roll = []
    pitch = []
    yaw = []
    for i in range(len(t)):   
        roll_,pitch_,yaw_ = quat_to_XYZ([qr[i],qx[i],qy[i],qz[i]])
        roll.append(roll_*TODEG)
        pitch.append(pitch_*TODEG)
        yaw.append(yaw_*TODEG)
        
    beta = []
    alpha = []
    gamma = []
    for i in range(len(t)):   
        beta_,alpha_,gamma_ = Quat.to_ZYZ([qr[i],qx[i],qy[i],qz[i]])
        beta.append(beta_*TODEG)
        alpha.append(alpha_*TODEG)
        gamma.append(gamma_*TODEG)
    
    plt.clf()
#    plt.title('ang')
    subplot(4612); plt.plot(t, x.T[iX],'r'); plt.title(r'$x^E$')
    subplot(4622); plt.plot(t, x.T[iY],'g'); 
    subplot(4632); plt.plot(t, x.T[iZ],'b'); 
    
    subplot(4611); plt.plot(t, eta.T[iVX],'r'); plt.title(r'$v^B$')
    subplot(4621); plt.plot(t, eta.T[iVY],'g'); 
    subplot(4631); plt.plot(t, eta.T[iVZ],'b'); 
    
    subplot(4613); plt.plot(t, eta.T[iWX],'r'); plt.title(r'$\omega_B$')
    subplot(4623); plt.plot(t, eta.T[iWY],'g'); 
    subplot(4633); plt.plot(t, eta.T[iWZ],'b'); 
   
    subplot(4644); plt.plot(t, qr,'k'); plt.ylim([-1.1, 1.1])
    subplot(4614); plt.plot(t, qx,'r'); plt.ylim([-1.1, 1.1]); plt.title(r'$q = q_{xy} \times q_{z}$')
    subplot(4624); plt.plot(t, qy,'g'); plt.ylim([-1.1, 1.1])
    subplot(4634); plt.plot(t, qz,'b'); plt.ylim([-1.1, 1.1])
    
    subplot(4645); plt.plot(t, qa[0],'k'); plt.ylim([-1.1, 1.1])
    subplot(4615); plt.plot(t, qa[1],'r'); plt.ylim([-1.1, 1.1]); plt.title(r'$q_{xy}$')
    subplot(4625); plt.plot(t, qa[2],'g'); plt.ylim([-1.1, 1.1])#plt.lim([-np.pi, np.pi])
    subplot(4635); plt.plot(t, qa[3],'b'); plt.ylim([-1.1, 1.1])#plt.lim([-np.pi, np.pi])
    
    subplot(4646); plt.plot(t, qb[0],'k'); plt.ylim([-1.1, 1.1])
    subplot(4616); plt.plot(t, qb[1],'r'); plt.ylim([-1.1, 1.1]); plt.title(r'$q_{z}$')
    subplot(4626); plt.plot(t, qb[2],'g'); plt.ylim([-1.1, 1.1])#plt.lim([-np.pi, np.pi])
    subplot(4636); plt.plot(t, qb[3],'b'); plt.ylim([-1.1, 1.1])#plt.lim([-np.pi, np.pi])
    
    roll = []
    pitch = []
    yaw = []
    for i in range(len(t)):   
        roll_,pitch_,yaw_ = quat_to_XYZ([qr[i],qx[i],qy[i],qz[i]])
        roll.append(roll_*TODEG)
        pitch.append(pitch_*TODEG)
        yaw.append(yaw_*TODEG)
        
    subplot(4641); plt.title('roll'); plt.plot(t, roll,'r');
    subplot(4642); plt.title('pitch'); plt.plot(t, pitch,'g');
    subplot(4643); plt.title('yaw'); plt.plot(t, yaw,'b');
#    plt.title('acc')
    if tight:
        plt.tight_layout()

        
def plot_F(t, torques, forces = None, tight=True, torque_lim = None, force_lim = None):
#    if len(F) != len(names):
#        raise ValueError('number of elements do not match')
        
    
    plt.clf()    

    if forces is None:
        m = 1
    else:
        m = 2
        names = []
        for f in forces:
            subplot(3,2,1,2); plt.plot(t, f[0].T[0]); 
            if force_lim is not None: plt.ylim(force_lim[0]);
            subplot(3,2,2,2); plt.plot(t, f[0].T[1]); 
            if force_lim is not None: plt.ylim(force_lim[1]);
            subplot(3,2,3,2); plt.plot(t, f[0].T[2]);
            if force_lim is not None: plt.ylim(force_lim[2]);
            names.append(f[1])
        plt.legend(names)
        subplot(3,2,1,2); plt.title('forces')
    
    names = []
    for f in torques:
        subplot(3,m,1,1); plt.plot(t, f[0].T[0]); 
        if torque_lim is not None: plt.ylim(torque_lim[0]);
        subplot(3,m,2,1); plt.plot(t, f[0].T[1]); 
        if torque_lim is not None: plt.ylim(torque_lim[1]);
        subplot(3,m,3,1); plt.plot(t, f[0].T[2]);
        if torque_lim is not None: plt.ylim(torque_lim[2]);
        names.append(f[1])
    plt.legend(names)
    subplot(3,m,1,1); plt.title('torques')
    
#    subplot(3121); plt.legend(names)
#    subplot(3131); plt.legend(names)
    
    if tight:
        plt.tight_layout()
        
    