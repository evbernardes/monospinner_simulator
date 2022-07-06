import bpy  
from mathutils import Vector, Quaternion
import json
import sys
import numpy as np
from math import sin, cos, sqrt, pi

def play(frame_step = 1, show_pos = False, frames_max = 1000):
    
    frames_max = 1000
    step = int(N / frames_max)
    
    data = {}
    data['t'] = list(t[::step])
    data['dt'] = DT
    data['tmax'] = tmax
    data['x'] = list(pos.T[0])
    data['y'] = list(pos.T[1])
    data['z'] = list(pos.T[2])
    data['show_pos'] = int(show_pos)
    data['qr'] = list(q.real[::step])
    data['qx'] = list(q.x[::step])
    data['qy'] = list(q.y[::step])
    data['qz'] = list(q.z[::step])
    data['gamma'] = list(angles.T[2][::step])
    #data['gamma'] = list(0*angles.T[2][::step])
    data['alpha'] = list(angles.T[1][::step])
    data['beta'] = list(angles.T[0][::step])
    data['frame_step'] = frame_step
    
    with open('.blender_temp.json', 'w') as outfile:
        json.dump(data, outfile)
    
    os.system('blender --python blender_test.py')

#%% get data
with open('sim.json') as json_file:
    data = json.load(json_file)
    dt = data['dt']
    tmax = data['tmax']
    t = data['t']
    x = data['x']
    y = data['y']
    z = data['z']
    show_pos = data['show_pos']
    qr = data['qr']
    qx = data['qx']
    qy = data['qy']
    qz = data['qz']
    gamma = data['gamma']
    alpha = data['alpha']
    beta = data['beta']
    frame_step = data['frame_step']

#%%
scale = 0.02
scale_ref = 0.9
#frame_step = 1
h = 124
rv = Vector([0,0,h])*scale
r = Quaternion([0,0,0,h])*scale
q0 = Quaternion([1,0,0],pi/2)

try:
    bpy.context.user_preferences.view.show_splash = False
except:
    try:
        bpy.types.PreferencesView.show_splash = False
    except:
        pass
#bpy.types.PreferencesView.show_splash = False

# useful shortcut
scene = bpy.context.scene
collection = bpy.context.collection

# clear everything for now
scene.camera = None  
for obj in scene.objects:  
    try:
        scene.objects.unlink(obj)
    except:
        collection.objects.unlink(obj)

stlfiles = [
            {'name':'body.stl'},
            {'name':'rotor_base.stl'},
            {'name':'thrust.stl'},
            {'name':'thrust_ref.stl'},
            {'name':'helix_centered.stl'}
            ]

# create sphere and make it smooth
# bpy.ops.import_mesh.stl(filepath=".", filter_glob="*.stl",  files=[{"name":"body.stl"}], directory="./3d files")
bpy.ops.import_mesh.stl(filepath=".", files=stlfiles, directory="./3d files")

# objects = []
objects = bpy.data.objects
print(objects)
for obj in bpy.data.objects:
    # obj.shade_smooth()
    obj.rotation_mode = 'QUATERNION'
    if obj.name != 'helix_centered':
        obj.rotation_quaternion = q0
    if obj.name == 'thrust_ref':
        obj.scale = [scale*scale_ref]*3
    else:
        obj.scale = [scale]*3
    # print(obj)

objects_centered = ['thrust','thrust_ref','helix_centered']
for obj_name in objects_centered:
    try:
        objects[obj_name].location = r[1:]
    except:
        pass

try:
    mat = bpy.data.materials.new("thrust_red")
    mat.diffuse_color = (1,0,0,1)
    try:
        bpy.data.objects['thrust'].data.materials.append(mat)
    except:
        pass
    try:
        bpy.data.objects['helix_centered'].data.materials.append(mat)
    except:
        pass
    
    mat = bpy.data.materials.new("thrust_ref_blue")
    mat.diffuse_color = (0,0,1,0.5)
    try:
        bpy.data.objects['thrust_ref'].data.materials.append(mat)
    except:
        pass
except:
    pass

qBR = Quaternion([0.,0.,1.],gamma[0])

## animation

# start with frame 0
number_of_frame = 1  
for xi,yi,zi,qri,qxi,qyi,qzi,gammai,alphai,betai,ti in zip(x,y,z,qr,qx,qy,qz,gamma,alpha,beta,t):
    
#    qri,qxi,qyi,qzi = 1.,0.,0.,0.
#    alphai = 0.
    
    qEB = Quaternion([qri,qxi,qyi,qzi]) # body's rotation
    qBR_ = Quaternion([0.,0.,1.],gammai) # rotor base rotationq
    qBR.conjugate()
    if((qBR @ qBR_)[0] < 0):
        qBR = -qBR_
    else:
        qBR = qBR_
#    qBR = qBR_
    
#    qBR = Quaternion([np.cos(gammai/2),0,0,np.sin(gammai/2)]) # rotor base rotation
    qBP = Quaternion([0,0,1],betai) @ Quaternion([0,1,0],alphai) # thrust vector rotation
    qFLAP = Quaternion([0,1,0],alphai*cos(gammai-betai)) # flapping rotation

    # now we will describe frame with number $number_of_frame
    scene.frame_set(number_of_frame)
    
    objects['body'].rotation_quaternion = q0
#    objects['body'].rotation_quaternion = qEB
    objects['rotor_base'].rotation_quaternion = qBR @ q0
    objects['thrust_ref'].rotation_quaternion = q0
    objects['thrust'].rotation_quaternion = qBP @ q0
    objects['helix_centered'].rotation_quaternion = qBR @ qFLAP
#    objects['helix_centered'].rotation_quaternion = qBR# @ qFLAP
    
    if show_pos:
        pos = Vector([xi,yi,zi])*scale
    else:
        pos = Vector([0,0,0])
    
    for obj in objects:
        obj.location = pos
        obj.rotation_quaternion = qEB @ obj.rotation_quaternion
        obj.keyframe_insert(data_path="rotation_quaternion", frame=number_of_frame, index=-1)
        
    for name in objects_centered:
        objects[name].location += qEB @ rv
        objects[name].keyframe_insert(data_path="location", index=-1)

    # move some frames
    number_of_frame += frame_step
scene.frame_end = number_of_frame