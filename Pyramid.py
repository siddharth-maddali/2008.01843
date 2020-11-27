import numpy as np
import functools as ftls


def distanceFromPlane( x, plane ):
    """
    INPUTS: 
    plane:   a 3x3 matrix whose columns are any three non-collinear 
        points on the plane
    x:       a 3xN array whose columns are the points whose signed 
        distances from the plane are required.
        
    OUTPUTS:
    Signed, perpendicular distances of each query point from the plane.
    The sign of each distance is determined by the conventional normal 
    to the plane.
    """
    planeNormal = ftls.reduce(
        lambda x, y: x + y, 
        [ np.cross( plane[:,n], plane[:,(n+1)%3] ) for n in [ 0, 1, 2 ] ]
    ).reshape( -1, 1 ) # right-handed normal computed
    planeNormal = planeNormal / np.linalg.norm( planeNormal )
    xShifted = x - plane[:,0].reshape( -1, 1 ).repeat( x.shape[1], axis=1 )
    return planeNormal.T @ xShifted

def onSameSideOfPlane( xQuery, plane, xTest ):
    """
    INPUTS:
    xTest:  a 3x1 array containing the test point in 3D.
    plane:  3x3 matrix whose columns denote non-collinear points in a plane.
    xQuery: a 3xN array containing query points in 3D
    
    OUPUTS:
    Boolean array, True for points in xQuery that are on the same side 
    of the plane as xTest, False otherwise.
    """
    dTest = distanceFromPlane( xTest, plane )[0][0] 
    dQuery = distanceFromPlane( xQuery, plane )
    return np.sign( dQuery )==np.sign( dTest )

def getPyramid( pyramidNodes, grid, finalShape, orig ):
    """
    INPUTS: 
    pyramidNodes:   3xN array denoting the 3D vertices of the pyramid.
    grid:           3xN array denoting uniformly sampled Cartesian coordinates in 3D space.
    finalShape:     List or tuple ( n1, n2, n3 ) denoting the pixel size of the final array.
                    Note: n1*n2*n3 = N should be satisfied, where N is the number of columns in xSample.
    orig:           3x1 array denoting the centroid of the pyramid. 
    
    OUTPUTS:
    pyramid:      Binary float array (0. or 1., denoting outside and inside of the pyramid respectively).
    """
    nodeLists = [ [4,0,1], [4,1,2], [4,2,3], [4,3,0], [0,1,2] ]
    pyramid = ftls.reduce( 
        lambda x, y: np.logical_and( x, y ), 
        [ 
            onSameSideOfPlane( grid, pyramidNodes[:,lst], orig ) 
            for lst in nodeLists 
        ]
    ).reshape( finalShape ).astype( float )
    return pyramid
