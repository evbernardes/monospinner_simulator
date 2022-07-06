#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:00:23 2021

@author: evbernardes
"""
import csv
import numpy as np

# math helpers
TODEG = 180 / np.pi
TORAD = 1/TODEG
Id = np.eye(3)
zeroM = np.zeros([3,3])
zeroV = np.zeros(3)
ex = np.array([1,0,0])
ey = np.array([0,1,0])
ez = np.array([0,0,1])
eps = 1E-5

def curvature(f, dt):
    try:
        x, y, z = f
    except:
        x, y, z = f.T
    
#    dt = np.diff(t)[0]
    dx_dt = np.diff(x)/dt
    dy_dt = np.diff(y)/dt
    
    dx_dt2 = np.diff(dx_dt)/dt
    dy_dt2 = np.diff(dy_dt)/dt
    
    k = np.abs(dx_dt2 * dy_dt[1:] - dy_dt2 * dx_dt[1:]) / ((dx_dt[1:]**2 + dy_dt[1:]**2 )**(3/2))
    k = np.append(k[0], k)
    return np.append(k[0], k)

def circle(r = 1):
    circle = np.array(range(360))*TORAD
    return [r*np.cos(circle), r*np.sin(circle)]

def cross(v):
    return np.array([[0,-v[2],v[1]],
                     [v[2],0,-v[0]],
                     [-v[1],v[0],0]])
Ex = cross(ex)
Ey = cross(ey)
Ez = cross(ez)

def ad(w, v):
    what = cross(w)
    vhat = cross(v)
    return mstack([[what, zeroM],[vhat, what]])

def get_middle_vector(q, k_old = None, e = np.array([0, 0, 1]), eps = 1E-5):
    k = q.rotate(e) + e
    N = np.linalg.norm(k)
    
    if N < eps:
        if k_old is None:
            return np.array([0, 1, 0])
        else:
            k = k_old - (k_old.dot(e))*e
            return k / np.linalg.norm(k)
    else:    
        return k / N

def find_n(a,b):
    diff = abs(a - b)
    return np.where(diff == min(diff))[0][0]

def find_n_max(a):
    return find_n(a,max(a))

def angdiff(x,y):
    S = np.sin(x)*np.cos(y) - np.cos(x)*np.sin(y)
    C = np.cos(x)*np.cos(y) + np.sin(x)*np.sin(y)
    return np.arctan2(S,C)

def angmean(x):
    S = np.mean(np.sin(x))
    C = np.mean(np.cos(x))
    return np.arctan2(S,C)

def wrap(phases, ang = np.pi):
    return (phases + ang) % (2 * ang) - ang

def unwrap_deg(x,discount):
    return (180/np.pi)*np.unwrap(x*np.pi/180,discount*np.pi/180)

def sg(y, window_size=31, order=1, deriv=0, rate=1):
     import numpy as np
     from math import factorial
     
     try:
         window_size = np.abs(np.int(window_size))
         order = np.abs(np.int(order))
     except (ValueError, msg):
         raise ValueError("window_size and order have to be of type int")
     if window_size % 2 != 1 or window_size < 1:
         raise TypeError("window_size size must be a positive odd number")
     if window_size < order + 2:
         raise TypeError("window_size is too small for the polynomials order")
     order_range = range(order+1)
     half_window = (window_size -1) // 2
     # precompute coefficients
     b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
     m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
     # pad the signal at the extremes with
     # values taken from the signal itself
     firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
     lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
     y = np.concatenate((firstvals, y, lastvals))
     return np.convolve( m[::-1], y, mode='valid')

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
    
def quaternion_divide_idx(quaternion,idx):
    n,m = quaternion.shape
    if m != 4:
        raise ValueError('quaternion array has wrong shape')
    
    q0 = np.mean(quaternion[idx],axis=0)
    print(q0)
    q0 = -q0/np.linalg.norm(q0)
    q0[0] = -q0[0]
    q = np.empty((n,m))
    for i in range(n):
        q[i] = quaternion_multiply(quaternion[i],q0)
    return q

# return handle for polynomial function of desired order
def poly_fun(pol_order,name):
    func_string = ''
    param_string = ''
    for i in range(pol_order+1):
        if i != 0:
            param_string = param_string+','
            func_string = func_string+' + '
        func_string = func_string+'a'+str(i)+'*x**'+str(i)
        param_string = param_string+'a'+str(i)
    func_string = f'def {name}(x,'+param_string+'): return '+func_string
    return func_string
#    exec(func_string)
#    return f
    
def normalize(v, eps = 10E-5):
    N = np.linalg.norm(v)
    if N < eps:
        return np.zeros(v.shape)
    else:
        return v/np.linalg.norm(v)

def mstack(matrices):
    N = len(matrices)
    M = len(matrices[0])
    for line in matrices[1:]:
        if len(line) != M:
            raise ValueError('lines must have the same number of matrices')
            
    lines = [np.hstack(matrices[i]) for i in range(N)]
    return np.vstack(lines)

def ZYZ_from_n(n):
    
    alpha = np.arccos(n[2])
    beta = np.arctan2(n[1], n[0])
    
    if np.isnan(alpha):
        alpha = 0
    if np.isnan(beta):
        beta = 0

    return [beta, alpha]

def R_from_ZYZ(beta,alpha,gamma):
    s1,c1 = np.sin(beta),np.cos(beta)
    s2,c2 = np.sin(alpha),np.cos(alpha)
    s3,c3 = np.sin(gamma),np.cos(gamma)
    return np.array([
            [- s3*s1 + c3*c2*c1, - c3*s1 - s3*c2*c1, s2*c1],
            [+ s3*c1 + c3*c2*s1, + c3*c1 - s3*c2*s1, s2*s1],
            [- c3*s2           ,              s3*s2,    c2]])
    
def quat_to_ZYZ(qr,qx,qy,qz,err = 1E-5):

    c1s2 = 2*(qx*qz+qr*qy)
    s1s2 = 2*(qy*qz-qr*qx)
    c2 = 2*(qr*qr+qz*qz)-1
    
    angle_1 = np.arctan2(s1s2,c1s2)
    
    angle_2 = np.empty(len(angle_1))
    for i in range(len(angle_1)):
        s1 = np.sin(angle_1[i])
        c1 = np.cos(angle_1[i])
        if np.abs(s1) > np.abs(c1):
            angle_2[i] = np.arctan2(s1s2[i]/s1,c2[i])
        else:
            angle_2[i] = np.arctan2(c1s2[i]/c1,c2[i])
            
#    angle_2 = np.arccos(c2); s2 = np.sign(np.sin(angle_2)); 
    
    s2s3 = 2*(qy*qz+qr*qx)
    s2c3 = -2*(qx*qz-qr*qy)
    angle_3 = np.arctan2(s2s3,s2c3)
    
#    idx = np.abs(angle_2) == 0
#    angle_3[idx] =+ angle_1[idx]
#    angle_1[idx] = 0
    idx = abs(angle_1) < err
    angle_3[idx] =+ angle_1[idx]
    angle_1[idx] = 0
    return [angle_1,angle_2,angle_3]
    
    
    return np.array([qr,qx,qy,qz])
    
def quat_to_R(q):
    qr,qx,qy,qz = q
    qv = np.array([[qx,qy,qz]]).T
    return (qr*qr - (qv.T @ qv)[0][0])*np.eye(3) + 2 * qv * qv.T + 2*qr*cross(qv.T[0])
