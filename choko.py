import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
dt = 1/240  
th0 = 0.1 
thd = 1.0  
jIdx = 1  
T = 2  
L = 0.5  
g = 10  
m = 1 
k = 0.5  
kp = 20
kd = 10
maxTime = 2*T 
logTime = np.arange(0.0, maxTime, dt)  
sz = len(logTime)
logPos = np.zeros(sz)  
logVel = np.zeros(sz)  
logCtrl = np.zeros(sz) 
idx = 0
tau = 0
physicsClient = p.connect(p.DIRECT)  
p.setGravity(0, 0, -10) 
boxId = p.loadURDF("./pendulum.urdf", useFixedBase=True)  
p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)

p.setJointMotorControl2(bodyIndex=boxId, jointIndex=jIdx, targetPosition=th0, controlMode=p.POSITION_CONTROL)
for _ in range(1000):
    p.stepSimulation()

p.setJointMotorControl2(bodyIndex=boxId, jointIndex=jIdx, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)
a3 = 10/T**3
a4 = -15/T**4
a5 = 6/T**5

a = np.array([0, 0, 0, a3, a4, a5])

for t in logTime[1:]:
    jointState = p.getJointState(boxId, jIdx)
    th1 = jointState[0]
    dth1 = jointState[1]
    if t < T:
        
        #th_d, thd_d, thdd_d = np.polyval(a, t), np.polyval(np.polyder(a), t), np.polyval(np.polyder(a, 2), t)
        th_d = a3*t**3 + a4*t**4 + a5*t**5
        thd_d = 3.0*a3*t**2 + 4.0*a4*t**3 + 5.0*a5*t**4
        thdd_d = 6.0*a3*t + 12.0*a4*t**2 + 20.0*a5*t**3
        error = th1 - th_d
        error_dot = dth1 - thd_d
        u = -kp*error - kd*error_dot + thdd_d
        
        tau = m*g*L*np.sin(th1) + k*dth1 + m*L*L*(u)

    p.setJointMotorControl2(
        bodyIndex=boxId,
        jointIndex=jIdx,
        controlMode=p.TORQUE_CONTROL,
        force=tau 
    )

    p.stepSimulation()

    logPos[idx] = th1
    logVel[idx] = dth1
    logCtrl[idx] = tau
    idx += 1

 
plt.subplot(3, 1, 1)
plt.grid(True)
plt.plot(logTime, logPos, label="simPos")
plt.plot([logTime[0], logTime[-1]], [thd, thd], 'r', label='refPos')
plt.legend()

plt.subplot(3, 1, 2)
plt.grid(True)
plt.plot(logTime, logVel, label="simVel")
plt.legend()

plt.subplot(3, 1, 3)
plt.grid(True)
plt.plot(logTime, logCtrl, label="simCtrl")
plt.legend()

plt.show()

p.disconnect()