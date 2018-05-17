# Copyright (c) 2016-2018, The University of Texas at Austin & University of
# California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

import dolfin as dl
import numpy as np

from petsc4py import PETSc

def amg_method():
    """
    Determine which AMG preconditioner to use.
    If avaialable use ML, which is faster than the PETSc one.
    """
    for pp in dl.krylov_solver_preconditioners():
        if pp[0] == 'ml_amg':
            return 'ml_amg'
        
    return 'petsc_amg'

def get_local_size(v):
    return v.get_local().shape[0]

def MatMatMult(A,B):
    """
    Compute the matrix-matrix product A*B.
    """
    Amat = dl.as_backend_type(A).mat()
    Bmat = dl.as_backend_type(B).mat()
    out = Amat.matMult(Bmat)
    rmap, _ = Amat.getLGMap()
    _, cmap = Bmat.getLGMap()
    out.setLGMap(rmap, cmap)
    return dl.Matrix(dl.PETScMatrix(out))

def MatPtAP(A,P):
    """
    Compute the triple matrix product P^T*A*P.
    """
    Amat = dl.as_backend_type(A).mat()
    Pmat = dl.as_backend_type(P).mat()
    out = Amat.PtAP(Pmat, fill=1.0)
    _, out_map = Pmat.getLGMap()
    out.setLGMap(out_map, out_map)
    return dl.Matrix(dl.PETScMatrix(out))

def MatAtB(A,B):
    """
    Compute the matrix-matrix product A^T*B.
    """
    Amat = dl.as_backend_type(A).mat()
    Bmat = dl.as_backend_type(B).mat()
    out = Amat.transposeMatMult(Bmat)
    _, rmap = Amat.getLGMap()
    _, cmap = Bmat.getLGMap()
    out.setLGMap(rmap, cmap)
    return dl.Matrix(dl.PETScMatrix(out))

def Transpose(A):
    """
    Compute the matrix transpose
    """
    Amat = dl.as_backend_type(A).mat()
    AT = PETSc.Mat()
    Amat.transpose(AT)
    rmap, cmap = Amat.getLGMap()
    AT.setLGMap(cmap, rmap)
    return dl.Matrix( dl.PETScMatrix(AT) )
    

def to_dense(A):
    """
    Convert a sparse matrix A to dense.
    For debugging only.
    """
    if hasattr(A, "getrow"):
        n  = A.size(0)
        m  = A.size(1)
        B = np.zeros( (n,m), dtype=np.float64)
        for i in range(0,n):
            [j, val] = A.getrow(i)
            B[i,j] = val
        
        return B
    else:
        x = dl.Vector()
        Ax = dl.Vector()
        A.init_vector(x,1)
        A.init_vector(Ax,0)
        
        n = get_local_size(Ax)
        m = get_local_size(x)
        B = np.zeros( (n,m), dtype=np.float64) 
        for i in range(0,m):
            i_ind = np.array([i], dtype=np.intc)
            x.set_local(np.ones(i_ind.shape), i_ind)
            A.mult(x,Ax)
            B[:,i] = Ax.get_local()
            x.set_local(np.zeros(i_ind.shape), i_ind)
            
        return B


def trace(A):
    """
    Compute the trace of a sparse matrix A.
    """
    n  = A.size(0)
    tr = 0.
    for i in range(0,n):
        [j, val] = A.getrow(i)
        tr += val[j == i]
    return tr

def get_diagonal(A, d, solve_mode=True):
    """
    Compute the diagonal of the square operator A
    or its inverse A^{-1} (if solve_mode=True).
    """
    ej, xj = dl.Vector(), dl.Vector()

    if hasattr(A, "init_vector"):
        A.init_vector(ej,1)
        A.init_vector(xj,0)
    else:       
        A.get_operator().init_vector(ej,1)
        A.get_operator().init_vector(xj,0)
        
    ncol = ej.size()
    da = np.zeros(ncol, dtype=ej.get_local().dtype)
    
    for j in range(ncol):
        ej[j] = 1.
        if solve_mode:
            A.solve(xj, ej)
        else:
            A.mult(ej,xj)
        da[j] = xj[j]
        ej[j] = 0.
        
    d.set_local(da)

      


def estimate_diagonal_inv2(Asolver, k, d):
    """
    An unbiased stochastic estimator for the diagonal of A^-1.
    d = [ \sum_{j=1}^k vj .* A^{-1} vj ] ./ [ \sum_{j=1}^k vj .* vj ]
    where
    - vj are i.i.d. ~ N(0, I)
    - .* and ./ represent the element-wise multiplication and division
      of vectors, respectively.
      
    REFERENCE:
    Costas Bekas, Effrosyni Kokiopoulou, and Yousef Saad,
    An estimator for the diagonal of a matrix,
    Applied Numerical Mathematics, 57 (2007), pp. 1214-1229.
    """
    x, b = dl.Vector(), dl.Vector()
    
    if hasattr(Asolver, "init_vector"):
        Asolver.init_vector(x,1)
        Asolver.init_vector(b,0)
    else:       
        Asolver.get_operator().init_vector(x,1)
        Asolver.get_operator().init_vector(b,0)
    
    num = np.zeros_like(b.get_local())
    den = np.zeros(num.shape, dtype = num.dtype)
    for i in range(k):
        x.zero()
        b_array = np.random.randn(num.shape[0])
        b.set_local(b_array)
        Asolver.solve(x,b)
        num = num +  ( x.get_local() * b_array )
        den = den +  (       b_array * b_array )
        
    d.set_local( num / den )
        
def randn_perturb(x, std_dev):
    """
    Add a Gaussian random perturbation to x:
    x = x + eta, eta ~ N(0, std_dev^2 I)
    """
    n = get_local_size(x)
    noise = np.random.normal(0, 1, n)
    x.set_local(x.get_local() + std_dev*noise)
    
class Solver2Operator:
    def __init__(self,S):
        self.S = S
        self.tmp = dl.Vector()
        
    def init_vector(self, x, dim):
        if hasattr(self.S, "init_vector"):
            self.S.init_vector(x,dim)
        elif hasattr(self.S, "operator"):
            self.S.operator().init_vector(x,dim)
        else:
            raise
        
    def mult(self,x,y):
        self.S.solve(y,x)
        
    def inner(self, x, y):
        self.S.solve(self.tmp,y)
        return self.tmp.inner(x)
    
class Operator2Solver:
    def __init__(self,op):
        self.op = op
        self.tmp = dl.Vector()
        
    def init_vector(self, x, dim):
        if hasattr(self.op, "init_vector"):
            self.op.init_vector(x,dim)
        else:
            raise
        
    def solve(self,y,x):
        self.op.mult(x,y)
        
    def inner(self, x, y):
        self.op.mult(y,self.tmp)
        return self.tmp.inner(x)
    
def vector2Function(x,Vh, **kwargs):
    """
    Wrap a finite element vector x into a finite element function in the space Vh.
    kwargs is optional keywords arguments to be passed to the construction of a dolfin Function
    """
    fun = dl.Function(Vh,**kwargs)
    fun.vector().zero()
    fun.vector().axpy(1., x)
    
    return fun
