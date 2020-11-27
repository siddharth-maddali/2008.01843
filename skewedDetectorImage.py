import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import functools as ftools

from argparse import Namespace
from matplotlib.colors import LogNorm
from scipy.spatial.transform import Rotation
from scipy.interpolate import griddata


################# USER EDIT ###########################

datafile = '../data/crystal_3.mat'
tilt_angles = [ deg*np.pi/180. for deg in [ 20., 0., 0. ] ] # in radians
image_samplers = np.array( 
    [ 
        [ 0.5, 0.25 ], 
        [ 0., 0.5 ]
    ]
)   # conveniently chosen numbers

#######################################################

data = Namespace( **sio.loadmat( datafile ) )

i, j = np.meshgrid( *[ np.arange( -n//2., n//2. ) for n in data.intens.shape[:-1] ] )
grid_coords = np.concatenate( 
    tuple( arr.reshape( -1, 1 ) for arr in [ i, j ] ), 
    axis=1 
)
sample_points = grid_coords @ image_samplers.T
points = np.array( ( i.flatten(), j.flatten() ) ).T

img = np.log10( data.intens[:,:,35]  )
img_interp = np.flipud( griddata( points, img.flatten(), sample_points ).reshape( 128, 128 ) )
sample_points += 64.

bounds = [ func( grid_coords ) for func in [ np.min, np.max ] ]
boundary = np.where(
    ftools.reduce( 
        np.logical_or, 
        [ list( map( lambda x: x in bounds, grid_coords[:,n] ) ) for n in [ 0, 1 ] ]
    )
)[0]

# plotting 
plt.close( 'all' )
plt.figure()
plt.imshow( data.intens[:,:,35], norm=LogNorm(), origin='lower' )
plt.set_cmap( 'gray' )
plt.plot( sample_points[boundary,0], sample_points[boundary,1], '.r' )
plt.xticks( [] )
plt.yticks( [] )

plt.figure()
plt.imshow( np.flipud( img_interp ), origin='lower' )
plt.xticks( [] )
plt.yticks( [] ) 


plt.show()
