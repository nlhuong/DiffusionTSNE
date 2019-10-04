## Data generation functions
import numpy as np

## Swiss Roll 3D -----------------------------------------------
def swiss_roll(n_points, noise, uniform = True, a = 1, b = 1.5):
    y = 15 * np.random.uniform(size = n_points) - 7.5
    if(uniform):
        t = np.sort(np.random.uniform(size = n_points))
    else:
        t = np.sort(np.random.beta(a, b, size=n_points))
    theta = (3 * np.pi / 2) * (1 + 2 * t)
    X = np.vstack((theta * np.cos(theta), y, theta * np.sin(theta)))
    X = np.transpose(X)
    E = noise * np.random.normal(size = X.shape)
    X += E
    # arc length: integral sqrt(1 + x^2) from 0 to theta 
    color = 0.5*(np.sqrt(1 + theta**2))*theta + np.arcsinh(theta)
    return X, color, y

## Spirals ------------------------------------------------------

def two_clusters(n1, n2, mu1=[-5.0, 0.0], mu2=[5, 0.0], 
                 s1=[1.0, 1.0], s2=[1.0, 1.0], 
                 density_evenness=.5, verbose=0):
    if not all([x == len(mu1) for x in [len(s1), len(mu2), len(s2)]]):
        raise ValueError("Lengths of mu1, mu2, s1, s2 must be equal")
    mu1 = np.array(mu1); mu2 = np.array(mu2)
    C1 = [np.random.normal(loc=mu1[i], scale=s1[i], size=n1) 
          for i in range(len(mu1))]
    C2 = [np.random.normal(loc=mu2[i], scale=s2[i], size=n2) 
          for i in range(len(mu2))]
    C1 = np.vstack(tuple(C1)).T
    C2 = np.vstack(tuple(C2)).T

    # Project on the line between two cluster centroids
    delta = mu2 - mu1
    delta_norm = np.sqrt(np.sum(delta**2))
    u = delta/delta_norm
    #Pu = np.outer(u, u.T)
    C1Proj = mu1 + np.outer(np.dot(C1 - mu1, u), u) 
    C2Proj = mu2 + np.outer(np.dot(C2 - mu2, u), u)

    # Further half of the clusters
    if density_evenness == 0.5:
        idx1 = np.sqrt(np.sum((C1Proj - mu2)**2, axis=1)) >= delta_norm
        idx2 = np.sqrt(np.sum((C2Proj - mu1)**2, axis=1)) >= delta_norm
        return(np.vstack((C1, C2)), np.hstack((np.zeros(n1), np.ones(n2))), 
               np.hstack((idx1, idx2)))
    elif density_evenness < 0.5:
        idx1 = np.sqrt(np.sum((C1Proj - mu2)**2, axis=1)) >= delta_norm
        idx2 = np.sqrt(np.sum((C2Proj - mu1)**2, axis=1)) >= delta_norm
    else:
        idx1 = np.sqrt(np.sum((C1Proj - mu2)**2, axis=1)) <= delta_norm
        idx2 = np.sqrt(np.sum((C2Proj - mu1)**2, axis=1)) <= delta_norm
        density_evenness = 1 - density_evenness
        
    idx1_flip = np.random.choice(
        np.where(idx1)[0], replace=False, size=int(sum(idx1) - n1*density_evenness))
    idx2_flip = np.random.choice(
        np.where(idx2)[0], replace=False, size=int(sum(idx2) - n2*density_evenness))

    C1[idx1_flip, :] += 2*(np.array(mu1) - C1Proj[idx1_flip, :])
    C2[idx2_flip, :] += 2*(np.array(mu2) - C2Proj[idx2_flip, :])
    C1Proj = mu1 + np.outer(np.dot(C1 - mu1, u), u) 
    C2Proj = mu2 + np.outer(np.dot(C2 - mu2, u), u)
    idx1 = np.sqrt(np.sum((C1Proj - mu2)**2, axis=1)) >= delta_norm
    idx2 = np.sqrt(np.sum((C2Proj - mu1)**2, axis=1)) >= delta_norm
    if verbose:
        frac = (sum(idx1) + sum(idx2))/(n1 + n2)
        print("Flipping %d points" %(len(idx1_flip) + len(idx2_flip)))
        print("Fraction of points in further halves is %f" %frac)
    return(np.vstack((C1, C2)), np.hstack((np.zeros(n1), np.ones(n2))), 
           np.hstack((idx1, idx2)))

                   

## Spirals ------------------------------------------------------

def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))))

def k_spirals(n_points, k, noise=.5, n_loops = 780 / 360,
             n=None):
    """
     Returns the two spirals dataset.
    """
    spirals = np.ndarray((k*n_points, 2))
    y = np.ndarray((k*n_points, ))
    if n is None:
        n = np.random.rand(n_points,1)
    #n = np.random.exponential(scale=1, size = (n_points,1)) 
    n *= (2*np.pi) * n_loops
    phase_shift = 2*np.pi/k
    for i in range(k):
        inoise = noise * (2*np.random.rand(n_points)-1) * phase_shift
        x1 = n*(-np.cos(n + i*phase_shift + inoise))
        x2 = n*(np.sin(n + i*phase_shift + inoise))
        spirals[range(i*n_points, (i+1)*n_points), 0] = x1.ravel()
        spirals[range(i*n_points, (i+1)*n_points), 1] = x2.ravel()
        y[range(i*n_points, (i+1)*n_points)] = i*np.ones(n_points)
    return(spirals, y)


def spiral3D(n_points, noise=.5, coils=3, r = 10, c = 1.5):
    t = np.linspace(0, 2*np.pi*coils, num=n_points).reshape((n_points, 1))
    X0 = np.hstack((r*np.cos(t), r*np.sin(t), c*t))
    E = noise*np.random.normal(size = X0.shape)
    s = np.sqrt(r**2 + c**2) * t.ravel()
    X = X0 + E
    return (X, s)