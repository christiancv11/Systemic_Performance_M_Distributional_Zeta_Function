
from itertools import *
import numpy as np
import numpy.linalg as ln
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
import h5py

N = 30
grafo = nx.barabasi_albert_graph( N, 2, seed = 1234 )
L = nx.laplacian_matrix( grafo ).todense()
links = list( nx.edges(grafo) )
degree = grafo.degree

def phiI( idx, args=() ):
    l, sigma = args
    return 2*np.pi*np.sum( np.sqrt(sigma/( sigma + l[idx] )) )

def nabla_rhoI( i, j, args=() ):
    l, sigma = args 

    if i == j: 
        return -np.pi*np.sqrt( sigma/(sigma + l[i])**3 )
    else:
        return 0.0

def phiII( idx, args = [] ):
    li, k, p, sigma = args
    return ( (2*np.pi)**N/( np.prod([ (1.0-sigma)*degree[i] + sigma*li[i] for i in idx ]) ) )**((-p+1)/(2*p))

def nabla_rhoII( i, j, args = [] ):
    li, n, p, sigma = args
    
    val = 0.0
    if i == j:
        return (-p+1)*sigma/(2*p)*( (2*np.pi)**n/( np.prod([ (1.0-sigma)*degree[i] + sigma*li[i] for i in range(2,n) ]) ) )**((-p+1)/(2*p))*( 1.0/((1.0 - sigma)*degree[i] + sigma*li[i]) )
    else:
        return 0.0

def iterate( nabla_rho, k, arg ):
    eps = set( combinations( range(grafo.number_of_nodes()), 2 ) )
    epsc = set( map( lambda x: tuple(x), links ) )
    ek = eps.difference( epsc )
    new_edges = []

    Lhat = np.zeros_like( L )

    for tau in range(k):
        a = []
        e_star = []
        ek = list( ek ) 
    
        for e in ek:
            a.append( nabla_rho( e[0], e[0], args = arg ) + nabla_rho( e[1], e[1], args=arg ) )

        idx = np.argmax( a )
        new_edges.append( ek[idx] )
        g = nx.Graph()
        g.add_nodes_from(range(0,N))
        g.add_edge( *ek[idx] )
        
        Lhat[:,:] += nx.laplacian_matrix(g)
        e_star = set([ ek[idx] ])        
        ek = set( ek )
        ek = ek.difference( e_star )

    return Lhat, new_edges

def efficiency_grow_links( phi, nabla_rho, arguments, Niter=28  ):
    N = grafo.number_of_nodes()
    efficience = []

    for k in range( Niter ):
        Lh, new_edges = iterate( nabla_rho, k, arguments )
        Ei, _ = ln.eigh(L+Lh)
        ei = Ei[np.argsort(Ei)]
        efficience.append( np.sum( phi( range(k+2,N), args=[ei,*arguments[1:]] ) ) )
    efficience = np.array( efficience )
    p0 = efficience[0]
    X = ( p0 - efficience )/p0*100
    return X, Lh, new_edges
	
def efficiency_original( phi, arguments ):
	N = grafo.number_of_nodes()
	efficience = []
	for k in range( N-2 ):
		efficience.append( np.sum( phi( range(k+2,N), args=arguments ) ) )
	efficience = np.array( efficience )
	p0 = efficience[0]
	X = ( p0 - efficience)/p0*100
	return X

if __name__ == "__main__":
    lambdai, _ = ln.eigh( L )
    li = lambdai[ np.argsort(lambdai) ]

    X, Lh, new_edges = efficiency_grow_links( phiI, nabla_rhoI, [li, 0.1], Niter = 10 )
    Y = efficiency_original( phiI, [li, 0.1] )

    # X, Lh, new_edges = efficiency_grow_links( phiII, nabla_rhoII, [li, N, 1.5, 0.1], Niter = 28 )
    # Y = efficiency_original( phiII, [li, N, 1.5, 0.1] )

    F = h5py.File("laplacian_matrix.h5","w")
    F.create_dataset( "/L", shape=(L.shape[0],L.shape[1]), dtype=int, data = L )
    F.create_dataset( "/Lh",shape=(L.shape[0],L.shape[1]), dtype=int, data = Lh )
    F.close()

    fig, ax = plt.subplots( 1, 1, figsize=(6, 6) )
    nx.draw_circular( grafo, ax=ax )

    G2 = nx.Graph()
    G2.add_nodes_from(range(N))
    G2.add_edges_from(new_edges)
    nx.draw_circular( G2, ax=ax, width=3, edge_color="r" )
    plt.savefig("modified_network.png")
