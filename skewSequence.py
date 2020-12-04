# Script to generate the image sequence of parameterized detector transformations

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

#################### user edit ################################

xi = 60. * np.pi/180.
zeta = 60. * np.pi/180.
phi = 73. * np.pi/180.

###############################################################

detframe = np.eye( 3 )

R1 = Rotation.from_rotvec( 
    xi * ( 
        np.cos( zeta )*detframe[:,0] + np.sin( zeta )*detframe[:,1] 
    ) 
).as_matrix()
R2 = Rotation.from_rotvec( phi * detframe[:,2] ).as_matrix()
#Rtilt = ( R2 * R1 ).as_matrix()

# generate points to be projected to image plane
pts = np.array( 
    [ 
        [ -1., -1. ], 
        [ -1., 1. ], 
        [ 1., 1. ], 
        [ 1., -1. ], 
        [ -1., -1. ]
    ]
)
pts = np.concatenate( ( pts, np.zeros( ( 5, 1 ) ) ), axis=1 ).T
pts1 = R1 @ pts
pts2 = R2 @ pts1


# plotting
plt.figure( 1 )
plt.plot( 0, 0, '.m' )
plt.plot( pts[0,:], pts[1,:], 'b', linewidth=4 )
plt.axis( 'equal' )
plt.axis( 'off' )
plt.savefig( '../indivFigures/sequence-1.pdf' )

plt.figure( 2 )
plt.plot( 0, 0, '.m' )
plt.plot( pts[0,:], pts[1,:], '-.b', linewidth=4 )
plt.plot( pts1[0,:], pts1[1,:], 'r', linewidth=4 )
plt.axis( 'equal' )
plt.axis( 'off' )
plt.savefig( '../indivFigures/sequence-2.pdf' )

plt.figure( 3 )
plt.plot( 0, 0, '.m' )
plt.plot( pts[0,:], pts[1,:], '-.b', linewidth=4 )
plt.plot( pts2[0,:], pts2[1,:], 'r', linewidth=4 )
plt.axis( 'equal' )
plt.axis( 'off' )
plt.savefig( '../indivFigures/sequence-3.pdf' )
