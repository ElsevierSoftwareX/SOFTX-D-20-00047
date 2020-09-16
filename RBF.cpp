/* LyapXool – V2: Eigenpairs, is a program to compute Complete Lyapunov functions,
 -> for dynamical systems described by non linear autonomous ordinary differential equations,
 ->
 ->
 -> This program is free software; you can redistribute it and/or
 -> modify it under the terms of the GNU General Public License
 -> as published by the Free Software Foundation; either version 3
 -> of the License, or (at your option) any later version.
 ->
 -> This program is distributed in the hope that it will be useful,
 -> but WITHOUT ANY WARRANTY; without even the implied warranty of
 -> MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 -> GNU General Public License for more details.
 ->
 -> You should have received a copy of the GNU General Public License
 -> along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ->
 -> Authors: Carlos Argáez, Peter Giesl, Sigurdur Freyr Hafstein
 */

#include <math.h>
#include <iostream>
#include <sstream>
#include <armadillo>
#include <fstream>
#include <list>
#include <string>
#include "RBF.hpp"
#include "instructions.hpp"
#include "odesystem.hpp"
#include "wendland.hpp"
#include <numeric>
#include "generalities.hpp"
#if defined(_OPENMP)
#include <omp.h>
#endif
using namespace arma;
using namespace std;
arma::span const All=span::all;

RBFMETHOD::RBFMETHOD(double alpha, int points_directional, double radius, int dimension, const double *min_geometric_limits, const double *max_geometric_limits, bool normal, bool printing, ofstream &outputf){
    this->alpha=alpha;
    this->points_directional=points_directional;
    this->radius=radius;
    this->dimension=dimension;
    this->min_geometric_limits=min_geometric_limits;
    this->max_geometric_limits=max_geometric_limits;
    this->normal=normal;
    this->printing=printing;
    this->outputf=&outputf;
    this->wdlfunction=0;
    this->wdlf1=0;
    this->wdlf2=0;
    this->wdlf3=0;
}
void RBFMETHOD::wbase()
{
    wall_clock timer;
    timer.tic();
    rbfbasis.set_size(dimension,dimension);
    
        rbfbasis.zeros();
        ek.set_size(dimension);
        ek.zeros();
        for(int k=1; k<=dimension; ++k)
        {
            
            ek(k-1)=sqrt(1.0/(2*k*(k+1)));
            rbfbasis(k-1,span::all)=ek;
            
            rbfbasis(k-1,All)(k-1)=(k+1)*ek(k-1);
        }
    
    if(printing)
    {
        printall("rbfbasis",normal,0,rbfbasis);
    }
    
    *outputf  << "Computing the base lasted: " << timer.toc() << " sec"<< endl;
    
    printhour(1);
}

void RBFMETHOD::alphafunction()
{
    
    wall_clock timer;
    timer.tic();
    alphasize=(int)collocationpoints.n_rows;
    alphavector.set_size(alphasize);
    alphavector.fill(-1.0);
    if(printing)
    {
        printall("alphavector", normal,  0, alphavector);
    }
    
    *outputf  << "The whole proceedure for the alpha vector lasted: " << timer.toc() << endl;
    
    printhour(1);
}



void RBFMETHOD::interpolationmatrixA(WENDLAND &wendland)
{
    wall_clock timer;
#if defined(_OPENMP)
    omp_set_dynamic(0);
    omp_set_num_threads(glovar::OMP_NUM_THREADS);
#endif
    int j=0, k=0;
    
    if(normal)
    {
        *outputf  << "Computing Interpolation Matrix with almost normalized function" << endl;
    }else{
        *outputf  << "Computing Interpolation Matrix with no normalized function" << endl;
    }
    
    double atzero;
    
    int dimA=(int)collocationpoints.n_rows;
    int dimAc=(int)collocationpoints.n_cols;
    
    *outputf  << "The length of the matrix is: " << dimA << endl;
    
    Amat.set_size(dimA,dimA);
    Amat.zeros();
    
    
    atzero=wendland.evawdlfn(0.0,*wdlf1);
    
    
    int chunk = int(floor(dimA/glovar::OMP_NUM_THREADS));
    
    timer.tic();
    
#pragma omp parallel shared(Amat,collocationpoints,wdlf1,wdlf2,atzero,normal,chunk) private(j,k)
    {
        rowvec diffsave(dimAc);
        rowvec savingcallj(dimAc);
        rowvec savingcallk(dimAc);
        rowvec resultj(dimAc);
        rowvec resultk(dimAc);
        
        diffsave.zeros();
        savingcallj.zeros();
        savingcallk.zeros();
        resultj.zeros();
        resultk.zeros();
        
        double twopointsdistance=0.0;
        
        double wdlfvalue1=0.0;
        double wdlfvalue2=0.0;
        double checking=0.0;
        
        
#pragma omp barrier
#pragma omp for schedule (dynamic, chunk) nowait
        for(j=0; j<dimA; ++j)
        {
            savingcallj=collocationpoints(j, All);
            odesystem(normal,savingcallj,resultj);
            for(k=0; k<dimA; ++k)
            {
                savingcallk=collocationpoints(k, All);
                diffsave=savingcallj-savingcallk;
                if(k==j){
                    Amat(j,k)=-atzero*dot(resultj,resultj);
                }else{
                    twopointsdistance=sqrt(dot(diffsave,diffsave));
                    checking=1.0-glovar::c*twopointsdistance;
                    if(checking>0.0)
                    {
                        odesystem(normal,savingcallk,resultk);
                        wdlfvalue1=wendland.evawdlfn(twopointsdistance,*wdlf1);
                        wdlfvalue2=wendland.evawdlfn(twopointsdistance,*wdlf2);
                        Amat(j,k)=-wdlfvalue2*dot(diffsave,resultj)*dot(diffsave,resultk)-wdlfvalue1*dot(resultj,resultk);
                    }
                }
            }
        }
    }
    
    *outputf  << "Computing the Interpolation matrix lasted: " << timer.toc() << " sec"<< endl;
    
    printhour(1);
}



void RBFMETHOD::choldecom()
{
    wall_clock timer;
    timer.tic();
    int maxite=(int)Amat.n_rows;
    R.set_size(maxite,maxite);
    R = chol(Amat);
    
    *outputf  << "The whole Cholesky proceedure last: " << timer.toc() << " sec"<< endl;
    printhour(1);
}


void RBFMETHOD::evaluatinggrid(int dimension, double cart_grid_density, mat &mutematrix)
{
    wall_clock timer;
    timer.tic();
    int elements;
    int mn=1;
    double maxmax=-INFINITY;
    double minmin=INFINITY;
    for(int jc=0; jc<dimension; ++jc)
    {
        if(min_geometric_limits[jc]<=minmin)
        {
            minmin=min_geometric_limits[jc];
        }
        if(max_geometric_limits[jc]>=maxmax)
        {
            maxmax=max_geometric_limits[jc];
        }
    }
    
    rowvec evaluatingpoints(1+(int)((abs(maxmax)+abs(minmin))/cart_grid_density));
    evaluatingpoints.zeros();//=0.0;
    for(int i=0; i<=(int)((abs(maxmax)+abs(minmin))/cart_grid_density); ++i)
    {
        evaluatingpoints(i)=minmin+i*cart_grid_density;
    }
    elements=(int)evaluatingpoints.size();
    vector<int> m(dimension);
    vector<vector<double>> v(dimension);
    for(auto i=v.begin(); i!=v.end();++i)
    {
        i->resize(elements);
        int kj=0;
        for(auto j=i->begin(); j!=i->end();++j)
        {
            *j=evaluatingpoints(kj);
            ++kj;
        }
    }
    for(int i=0; i<dimension; ++i)
    {
        m[i]=(int)v[i].size();
        mn*=m[i];
    }
    
    mutematrix.set_size(mn,dimension);
    for(int i=0; i<mn;++i){
        int k=i;
        for(int j=dimension-1;j>=0;--j){
            mutematrix(i,j)=v[j][k%m[j]];
            k/=m[j];
        }
    }
    *outputf << "The whole proceedure for the evaluating grid lasted: " << timer.toc() << " sec"<< endl;
}

void RBFMETHOD::effectivegridnewnew(mat &gridtobeclean)
{
    wall_clock timer;
    timer.tic();
    for(int jc=0; jc<dimension; ++jc)
    {
        if(max_geometric_limits[jc]<=min_geometric_limits[jc])
        {
            *outputf  << "ERROR: Maximum should be larger than Minimum" << endl;
            *outputf  << "Entry: " << jc << " value: " << max_geometric_limits[jc] << endl;
            finalization();
            exit(9);
        }
    }
    int dim1=(int)gridtobeclean.n_rows;//longitud
    int dim2=(int)gridtobeclean.n_cols;//anchura
    
    list<int> counter;
    for(int i=0; i<dim1; ++i)
    {
        int inside=0;
        for(int jc=0; jc<dimension; ++jc)
        {
            if((gridtobeclean(i,jc)<=max_geometric_limits[jc] && gridtobeclean(i,jc)>=min_geometric_limits[jc]))
            {
                ++inside;
            }else{
                break;
            }
        }
        if(inside==dimension)
        {
            counter.push_back(i);
        }
    }
    
    
    
    int fin=(int)counter.size();
    cartesianevalgrid.set_size(fin,dim2);
    int n=0;
    for(list<int>::iterator i=counter.begin(); i!=counter.end(); ++i)
    {
        cartesianevalgrid(n,All)=gridtobeclean(*i,All);
        n++;
    }
    counter.clear();
    
    if(printing)
    {
        printall("dense", normal,  0, cartesianevalgrid);
    }
    
    *outputf  << "The total amount of points in the Cartesian grid (the length of your domain) is: " << n << endl;
    *outputf << "Constraining the Cartesian grid to the boundaries lasted: " << timer.toc() << " sec"<< endl;
    printhour(1);
}


void RBFMETHOD::direcgrid()
{
    wall_clock timer;
    timer.tic();
    
    int lcols=(int)collocationpoints.n_cols;
    int lrows=(int)collocationpoints.n_rows;
    
    
    
    int newlenght=(int)(points_directional*2*lrows);
    
    
    int j,jd;
    stride=points_directional*2+1;
    
    double norm;
    coldirectgrid.set_size(lrows*stride,lcols);
    directgrid.set_size(newlenght,lcols);
    
    
    mat domain(newlenght,lcols);
    rowvec savingdomain(lcols), evaldfunction(lcols);
    {
        for(int i=0; i<lrows; ++i)
        {
            j=stride*i;
            jd=(stride-1)*i;
            coldirectgrid(j,All)=collocationpoints(i,All);
            savingdomain=collocationpoints(i,All);
            
            odesystem(normal, savingdomain, evaldfunction);
            norm=sqrt(dot(evaldfunction,evaldfunction));
            int kp=0;
            for(int kd=0; kd<points_directional; kd+=1)
            {
                directgrid(jd+kp,All)=collocationpoints(i,All)+(radius/points_directional)*(kd+1)*alpha*(evaldfunction/norm);
                directgrid(jd+kp+1,All)=collocationpoints(i,All)-(radius/points_directional)*(kd+1)*alpha*(evaldfunction/norm);
                coldirectgrid(j+kp+1,All)=directgrid(jd+kp,All);
                coldirectgrid(j+kp+2,All)=directgrid(jd+kp+1,All);
                kp+=2;
                
            }
        }
    }
    
    
    list<int> counter,counterf;
    {
        int cdrows=(int)coldirectgrid.n_rows;
        int drows=(int)directgrid.n_rows;
        for(int i=0; i<cdrows; ++i)
        {
            for(int jc=0; jc<dimension; ++jc)
            {
                if((coldirectgrid(i,jc)<=max_geometric_limits[jc]) && (coldirectgrid(i,jc)>=min_geometric_limits[jc]))
                {
                    counter.push_back(i);
                }
                
            }
        }
        for(int ii=0; ii<drows; ++ii)
        {
            for(int jc=0; jc<dimension; ++jc)
            {
                if((directgrid(ii,jc)<=max_geometric_limits[jc]) && (directgrid(ii,jc)>=min_geometric_limits[jc]))
                {
                    counterf.push_back(ii);
                }
            }
        }
    }
    
    int ana=(int)counter.size();
    int flo=(int)counterf.size();
    
    boolcoldirectgrid.resize(stride*lrows,false);
    booldirectgrid.resize(lrows*(stride-1),false);
    cleanbigag.set_size(ana,lcols);
    cleanbigfg.set_size(flo,lcols);
    
    int n=0;
    int m=0;
    
    for(list<int>::iterator i=counter.begin(); i!=counter.end(); ++i)
    {
        boolcoldirectgrid[*i]=true;
        cleanbigag(n,All)=coldirectgrid(*i,All);
        n++;
    }
    for(list<int>::iterator ii=counterf.begin(); ii!=counterf.end(); ++ii)
    {
        booldirectgrid[*ii]=true;
        cleanbigfg(m,All)=directgrid(*ii,All);
        m++;
    }
    
    counter.clear();
    counterf.clear();
    
    if(printing)
    {
        printall("direcgrid",normal, 0, directgrid);
    }
    
    
    *outputf  << "The total amount of points to evaluate your function (the length of your domain) is: " << directgrid.n_rows << endl;
    *outputf  << "The whole proceedure to construct the evaluation grid lasted: " << timer.toc() << " sec"<< endl;
}

void RBFMETHOD::makecolgrid()
{
    wall_clock timer;
    timer.tic();
    RBFMETHOD::wbase();
    
    arma::rowvec  a(min_geometric_limits, glovar::ode_dimension);
    arma::rowvec  b(max_geometric_limits, glovar::ode_dimension);
    double tol = 1e-10;
    a -= tol * ones<rowvec>(dimension);
    b += tol * ones<rowvec>(dimension);
    list<rowvec> Ret;
    function<void(int, rowvec)> ML = [&](int r, rowvec x) {
        for (int i = int(ceil((a(r) - x(r)) / (alpha*(r + 2)*ek(r)))); i <= int(floor((b(r) - x(r)) / (alpha*(r + 2)*ek(r)))); i++) {
            if (r == 0) {
                Ret.push_back(x + i *alpha*rbfbasis(r,All));
            }
            else {
                ML(r - 1, x + i *alpha*rbfbasis(r,All));
            }
        }
    };
    ML(dimension - 1, 0.5*alpha*solve(rbfbasis, ones<vec>(dimension)).t());

    function<bool(const rowvec &)> BadColl = [&](const rowvec &x)->bool {
        rowvec f(dimension);
        
        odesystem(false, x, f);
        return norm(f) < 1e-10;
    };
	Ret.remove_if(BadColl);

    int NrOfPoints = (int)Ret.size();
    collocationpoints.set_size(NrOfPoints, dimension);
    auto Ri = Ret.begin();
    for (int i = 0; i < NrOfPoints; Ri++, i++) {
        collocationpoints(i, All) = *Ri;
    }
    *outputf << "The total proceedure to construct the collocation points lasted: " << timer.toc() << " sec"<< endl;
}



void RBFMETHOD::makeRBF(WENDLAND &wendland)
{
    wendland.wendlandfunction();
    wendland.wendlandderivative(wendland.wdlfunction,wendland.wdlf1);
    wendland.wendlandderivative(wendland.wdlf1,wendland.wdlf2);
    wendland.wendlandderivative(wendland.wdlf2,wendland.wdlf3);
    wdlfunction=&wendland.wdlfunction;
    wdlf1=&wendland.wdlf1;
    wdlf2=&wendland.wdlf2;
    wdlf3=&wendland.wdlf3;
    makecolgrid();
    rbfbasis.clear();
    alphafunction();
    direcgrid();
    interpolationmatrixA(wendland);
    choldecom();
    Amat.clear();
}

