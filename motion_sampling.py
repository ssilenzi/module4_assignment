# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt

# Sample from a normal distribution using 12 uniform samples.
def sample_normal_distribution(mu, sigma):
  #return sample from normal distribution with mean = a and standard deviation b
  sum = 0
  for ii in range(0,12):
    sum += np.random.uniform(-sigma, sigma)
  return sum / 2 + mu 


  """ Sample odometry motion model.
  
  Arguments:
  x -- pose of the robot before moving [x, y, theta]
  u -- odometry reading obtained from the robot [rot1, rot2, trans]
  a -- noise parameters of the motion model [a1, a2, a3, a4]
  
  """
def sample_motion_model(x, u, a):
  rot1_prime = u[0] + sample_normal_distribution(0, a[0]*abs(u[0]) + a[1]*u[2])
  rot2_prime = u[1] + sample_normal_distribution(0, a[0]*abs(u[1]) + a[1]*u[2])
  trans_prime = u[2] + sample_normal_distribution(0, a[2]*u[2] + a[3]*(abs(u[0]) + abs(u[1])))
  x_prime = x[0] + trans_prime * math.cos(x[2] + rot1_prime)
  y_prime = x[1] + trans_prime * math.sin(x[2] + rot1_prime)
  theta_prime = x[2] + rot1_prime + rot2_prime
  return np.array([x_prime, y_prime, theta_prime])


""" Evaluate motion model """

def main():
  # start pose
  x = [0.0, 0.0, 0.0]
  # odometry
  u = [[0.0, 0.0, 1.0],
       [0.0, 0.0, 1.0],
       [np.pi/2, 0.0, 1.0],
       [0.0, 0.0, 1.0],
       [0.0, 0.0, 1.0],
       [np.pi/2, 0.0, 1.0],
       [0.0, 0.0, 1.0],
       [0.0, 0.0, 1.0],
       [0.0, 0.0, 1.0],
       [0.0, 0.0, 1.0]]
  # noise parameters
  a = [0.02, 0.02, 0.01, 0.01]

  num_samples = 1000
  x_prime = np.zeros([num_samples, 3])
  # 1000 samples with initial pose
  for i in range(0, num_samples):
      x_prime[i,:] = x

  plt.axes().set_aspect('equal')
  plt.xlim([-5, 5])
  plt.ylim([-3, 7])
  plt.plot(x[0], x[1], "bo")
  
  # incrementally apply motion model
  for i in range(0,10):
      for j in range(0, num_samples):
        x_prime[j,:] = sample_motion_model(x_prime[j,:],u[i],a)        
      plt.plot(x_prime[:,0], x_prime[:,1], "r,")

  
  plt.xlabel("x-position [m]")
  plt.ylabel("y-position [m]")

  plt.show()

if __name__ == "__main__":
  main()


