import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as F
from PIL import Image
import random
try:
    from scipy.special import comb
except:
    from scipy.misc import comb

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = torch.tensor([p[0] for p in points], dtype=torch.float32).cuda()
    yPoints = torch.tensor([p[1] for p in points], dtype=torch.float32).cuda()

    # t = torch.linspace(0.0, 1.0, nTimes,dtype=torch.float32).cuda()
    # xPoints = np.array([p[0] for p in points])
    # yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = torch.tensor([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)], dtype=torch.float32).cuda()

    xvals = torch.matmul(xPoints, polynomial_array)
    yvals = torch.matmul(yPoints, polynomial_array)

    return xvals, yvals


def nonlinear_transformation(x, prob=0.8,mode='weak'):          
    points = [[1, 1], [random.random(), random.random()], [random.random(), random.random()], [0,0]]
    xvals, yvals = bezier_curve(points, nTimes=100)
    xvals, yvals = torch.sort(xvals)[0], torch.sort(yvals)[0]
    
    if mode=='weak':
        if random.random() >= prob:
                return x
        nonlinear_x = np.interp(np.array(x.cpu()), np.array(xvals.cpu()), np.array(yvals.cpu()))
    elif mode=='strong':
        nonlinear_x = np.interp(np.array(x.cpu()), np.array(xvals.cpu()), np.array(1-yvals.cpu()))
    return torch.tensor(nonlinear_x).cuda()



def transforms_for_rot(ema_inputs):

    rot_mask = np.random.randint(0, 4, ema_inputs.shape[0])
    flip_mask = np.random.randint(0, 2, ema_inputs.shape[0])

    # flip_mask = [0,0,0,0,1,1,1,1]
    # rot_mask = [0,1,2,3,0,1,2,3]

    for idx in range(ema_inputs.shape[0]):
        if flip_mask[idx] == 1:
            ema_inputs[idx] = torch.flip(ema_inputs[idx], [1])

        ema_inputs[idx] = torch.rot90(ema_inputs[idx], int(rot_mask[idx]),dims=[1,2])

    return ema_inputs, rot_mask, flip_mask
