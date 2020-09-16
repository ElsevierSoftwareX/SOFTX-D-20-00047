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
#include "lyapunovfunction.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif
using namespace arma;
using namespace std;

arma::span const All=span::all;
LYAPUNOV::LYAPUNOV(int totaliterations, int ode_dimension, double cart_grid_density, int l, int k, double c, int points_directional, double critval, bool normal, bool printing, std::ofstream &outputf){
    this->totaliterations=totaliterations;
    this->ode_dimension=ode_dimension;
    this->cart_grid_density=cart_grid_density;
    this->l=l;
    this->k=k;
    this->c=c;
    this->points_directional=points_directional;
    this->critval=critval;
    this->normal=normal;
    this->printing=printing;
    this->outputf=&outputf;
}


void LYAPUNOV::lyapequation(int currentiteration, RBFMETHOD &rbf)
{
    
    wall_clock timer;
    int maxbet=(int)rbf.collocationpoints.n_rows;
    betaod.set_size(maxbet);
    
    timer.tic();
    
    betaod=solve(trimatu(rbf.R), solve(trimatl(trimatu(rbf.R).t()), rbf.alphavector),solve_opts::fast);
    
    *outputf  << "The whole proceedure solve the Lyapunov equation at iteration " << currentiteration << " lasted: " << timer.toc() << " sec"<< endl;
    if(printing)
    {
        printall("betavector", normal,  currentiteration, betaod);
    }
    
}

void LYAPUNOV::lyapunovfunctions(int currentiteration, bool type_of_grid, mat &evalcoordinates, WENDLAND &wendland, RBFMETHOD &rbf)
{
#if defined(_OPENMP)
    omp_set_dynamic(0);
    omp_set_num_threads(glovar::OMP_NUM_THREADS);
#endif
    int i=0, k=0;
    
    wall_clock timer;
    timer.tic();
    if(normal)
    {
        *outputf  << "Computing the Lyapunov function with almost normalized function" << endl;
    }else{
        *outputf  << "Computing the Lyapunov function with no normalized function" << endl;
    }
    
    int chunk;
    int maxite=(int)evalcoordinates.n_rows;
    int maxbet=(int)rbf.collocationpoints.n_rows;
    int pointdim=(int)rbf.collocationpoints.n_cols;
    chunk = int(floor(maxite/glovar::OMP_NUM_THREADS));
#pragma omp parallel num_threads(glovar::OMP_NUM_THREADS) shared(lyapfunc,orbder,chunk) private(i,k)
    {
        lyapfunc.set_size(maxite);
        orbder.set_size(maxite);
        lyapfunc.zeros();
        orbder.zeros();
        rowvec diffpoints(pointdim), diffpointski(pointdim), diffpointskineg(pointdim);
        rowvec resulti(pointdim), resultk(pointdim), saving(pointdim), savingdomain(pointdim);
        diffpoints.zeros();
        diffpointski.zeros();
        diffpointskineg.zeros();
        resulti.zeros();
        resultk.zeros();
        saving.zeros();
        savingdomain.zeros();
        
        double proctk=0.0;
        double producting=0.0;
        double twopointsdistance=0.0;
        double wdlfvalue1=0.0;
        double wdlfvalue2=0.0;
        double checking=0.0;
        
#pragma omp barrier
#pragma omp for schedule (dynamic, chunk)// nowait
        for(i=0; i<maxite; ++i)
        {
            savingdomain=evalcoordinates(i,All);
            odesystem(normal,savingdomain,resulti);
            for(k=0; k<maxbet; ++k)
            {
                saving=rbf.collocationpoints(k,All);
                diffpoints=savingdomain-saving;
                twopointsdistance=sqrt(dot(diffpoints,diffpoints));
                checking=1.0-c*twopointsdistance;
                if(checking>0.0)
                {
                    odesystem(normal,saving,resultk);
                    wdlfvalue1=wendland.evawdlfn(twopointsdistance, wendland.wdlf1);
                    wdlfvalue2=wendland.evawdlfn(twopointsdistance, wendland.wdlf2);
                    diffpointski=saving-savingdomain;
                    proctk=dot(diffpointski,resultk);
                    producting=betaod(k)*proctk;
                    lyapfunc(i)+=producting*wdlfvalue1;
                    orbder(i)+=-wdlfvalue2*producting*dot(diffpointski,resulti)-betaod(k)*wdlfvalue1*dot(resulti,resultk);
                }
            }
        }
    }
    
    if(printing)
    {
        if(type_of_grid)
        {
            printall("lyapfuncdir", normal,  currentiteration, lyapfunc);
            printall("orbderdir", normal,  currentiteration, orbder);
        }else{
            printall("lyapfunccar", normal,  currentiteration, lyapfunc);
            printall("orbdercar", normal,  currentiteration, orbder);
        }
    }
    
    
    *outputf  << "The whole proceedure to compute the Lyapunov function lasted: " << timer.toc() << " sec"<< endl;
    printhour(1);
}


void LYAPUNOV::firstderivative(int currentiteration, bool type_of_grid, mat &evalcoordinates, WENDLAND &wendland, RBFMETHOD &rbf)
{
#if defined(_OPENMP)
    omp_set_dynamic(0);
    omp_set_num_threads(glovar::OMP_NUM_THREADS);
#endif
    
    wall_clock timer;
    timer.tic();
    
    
    int i=0,j=0,k=0;
    int evaldim=(int)evalcoordinates.n_rows;
    
    fdvector.set_size(evaldim,ode_dimension);
    fdvector.zeros();
    
    double checking=0.0;
    double twopointsdistance=0.0;
    double wdlfvalue1=0.0;
    double wdlfvalue2=0.0;
    
    
    int chunk = int(floor(evaldim/glovar::OMP_NUM_THREADS));
#pragma omp parallel num_threads(glovar::OMP_NUM_THREADS) shared(fdvector,chunk) private(i,j,k,twopointsdistance,wdlfvalue1,wdlfvalue2,checking)
    {
        rowvec saving(ode_dimension), savingdomain(ode_dimension), diffpoints(ode_dimension),resultk(ode_dimension);
        int maxite=(int)betaod.n_rows;
#pragma omp barrier
#pragma omp for schedule (dynamic, chunk) nowait
        for(j=0; j<evaldim; ++j)
        {
            saving=evalcoordinates(j,All);
            for(i=0; i<ode_dimension; ++i)
            {
                for(k=0; k<maxite; ++k)
                {
                    savingdomain=rbf.collocationpoints(k, All);
                    odesystem(normal,savingdomain,resultk);
                    diffpoints=saving-savingdomain;
                    twopointsdistance=sqrt(dot(diffpoints,diffpoints));
                    checking=1.0-glovar::c*twopointsdistance;
                    if(checking>0.0)
                    {
                        wdlfvalue1=wendland.evawdlfn(twopointsdistance, wendland.wdlf1);
                        wdlfvalue2=wendland.evawdlfn(twopointsdistance, wendland.wdlf2);
                        fdvector(j,i)+=betaod(k)*(-resultk(i)*wdlfvalue1
                                                  -diffpoints(i)
                                                  *dot(diffpoints,resultk)
                                                  *wdlfvalue2);
                    }
                }
            }
        }
    }
    double numbernormsquare=0.0;
    normed.set_size(evaldim);
    for(int p=0; p<evaldim; ++p)
    {
        numbernormsquare=dot(fdvector(p,All),fdvector(p,All));
        normed(p)=sqrt(numbernormsquare);
    }
    
    if(printing)
    {
        if(type_of_grid)
        {
            printall("lyapprimexdir", normal,  currentiteration, fdvector);
            printall("normeddire", normal,  currentiteration, normed);
        }else{
            printall("lyapprimexcar", normal,  currentiteration, fdvector);
            printall("normedcar", normal,  currentiteration, normed);
        }
    }
    *outputf << "The whole proceedure to compute the gradient lasted: " << timer.toc() << " sec"<< endl;
    printhour(1);
}

void LYAPUNOV::secondderivative(mat &evalcoordinates, WENDLAND &wendland, RBFMETHOD &rbf)
{
#if defined(_OPENMP)
    omp_set_dynamic(0);
    omp_set_num_threads(glovar::OMP_NUM_THREADS);
#endif
    wall_clock timer;
    timer.tic();
    
    
    int i=0,j=0,k=0;
    int evaldim=(int)evalcoordinates.n_rows;
    
    sdvector.set_size(ode_dimension,ode_dimension,evaldim);
    sdvector.zeros();
    
    int chunk = int(floor(evaldim/glovar::OMP_NUM_THREADS));
#pragma omp parallel num_threads(glovar::OMP_NUM_THREADS) shared(sdvector,chunk) private(i,j,k)
    {
        rowvec resultk(ode_dimension), saving(ode_dimension), savingdomain(ode_dimension), diffpoints(ode_dimension);
        int maxite=(int)betaod.n_rows;
        double twopointsdistance=0.0;
        double wdlfvalue2=0.0;
        double wdlfvalue3=0.0;
        double kdelta=0;
#pragma omp barrier
#pragma omp for schedule (dynamic, chunk) nowait
        for(int p=0; p<evaldim; ++p)
        {
            savingdomain=evalcoordinates(p,All);//equis
            for(i=0; i<ode_dimension; ++i)
            {
                for(j=0; j<ode_dimension; ++j)
                {
                    if(i==j)
                    {
                        kdelta=1.0;
                    }else{
                        kdelta=0.0;
                    }
                    for(k=0; k<maxite; ++k)
                    {
                        saving=rbf.collocationpoints(k, All);//equisk
                        odesystem(normal,saving,resultk);
                        diffpoints=savingdomain-saving;
                        twopointsdistance=sqrt(dot(diffpoints,diffpoints));
                        checking=1.0-glovar::c*twopointsdistance;
                        
                        if(checking>0.0)
                        {
                            wdlfvalue2=wendland.evawdlfn(twopointsdistance,wendland.wdlf2);
                            wdlfvalue3=wendland.evawdlfn(twopointsdistance,wendland.wdlf3);
                            sdvector(i,j,p)+=betaod(k)*(
                                                        -diffpoints(j)*resultk(i)*wdlfvalue2
                                                        -kdelta*dot(diffpoints,resultk)*wdlfvalue2
                                                        -diffpoints(i)*resultk(j)*wdlfvalue2
                                                        -diffpoints(i)*diffpoints(j)*dot(diffpoints,resultk)*wdlfvalue3
                                                        );
                        }
                    }
                }
            }
        }
    }
    *outputf << "The whole proceedure to compute the Hessian lasted: " << timer.toc() << " sec"<< endl;
    printhour(1);
}


void LYAPUNOV::findingeigenamount(int currentiteration, bool type_of_grid, RBFMETHOD &rbf)
{
#if defined(_OPENMP)
    omp_set_dynamic(0);
    omp_set_num_threads(glovar::OMP_NUM_THREADS);
#endif
    wall_clock timer;
    timer.tic();
    
    
    
    
    int maxlim=(int)sdvector.n_slices;
    int dimlim=(int)rbf.collocationpoints.n_cols;
    
    mat x(dimlim,dimlim);
    TOTALEIGEN.set_size(maxlim,dimlim);
    TOTALEIGENV.set_size(dimlim,dimlim,maxlim);
    int i=0;
    cx_vec eigval;
    cx_mat eigvec;
    
    int chunk = int(floor(maxlim/glovar::OMP_NUM_THREADS));
    
#pragma omp parallel shared(TOTALEIGEN,chunk) private(i,x,eigval, eigvec)
    {
#pragma omp barrier
#pragma omp for schedule (dynamic, chunk) nowait
        for(i=0; i<maxlim; ++i)
        {
            x=sdvector(All,All,span(i));
            eig_gen(eigval, eigvec, x);
            TOTALEIGEN(i,All)=sort(real( eigval.t() ));
            TOTALEIGENV(span(0,dimlim-1),span(0,dimlim-1),span(i))=real(eigvec);
        }
    }
    
    if(printing)
    {
        
        if(type_of_grid)
        {
            printall("eigendir", normal,  currentiteration, TOTALEIGEN);
            printcube("eigenvecdir", normal,  currentiteration, TOTALEIGENV);
        }else{
            printall("eigencar", normal,  currentiteration, TOTALEIGEN);
            printcube("eigenveccar", normal,  currentiteration, TOTALEIGENV);
        }
    }
    *outputf << "The whole proceedure to compute the Eigenvalues lasted: " << timer.toc() << " sec"<< endl;
    printhour(1);
}


void LYAPUNOV::chainrecurrentset(int currentiteration, bool type_of_grid, bool with_orbder, mat &evalcoordinates)
{
    list<int> counterzero;
    int maxlength=(int)evalcoordinates.n_rows;
    int maxwidth=(int)evalcoordinates.n_cols;
    for(int j=0; j<maxlength; ++j)
    {
        if(with_orbder)
        {
            if(orbder(j)>critval)
            {
                counterzero.push_back(j);
            }
        }else{
            if(-normed(j)>-critval)
            {
                counterzero.push_back(j);
            }
        }
    }
    int faillength=(int)counterzero.size();
    crslyapun.set_size(faillength);
    crsorbder.set_size(faillength);
    failinggrid.set_size(faillength,maxwidth);
    failinglyapunov.set_size(faillength);
    failingorbder.set_size(faillength);
    int m=0;
    {
        for(list<int>::iterator ii=counterzero.begin(); ii!=counterzero.end(); ++ii)
        {
            crslyapun(m)=lyapfunc(*ii);
            crsorbder(m)=orbder(*ii);
            failinglyapunov(m)=lyapfunc((*ii));
            failingorbder(m)=orbder((*ii));
            failinggrid(m,All)=evalcoordinates(*ii,All);
            m++;
        }
    }
    counterzero.clear();
    if(printing)
    {
        if(type_of_grid)
        {
            printall("fdirecgrid", normal,  currentiteration, failinggrid);
            printall("flfdirecgrid", normal,  currentiteration, failinglyapunov);
            printall("flfpdirecgrid", normal,  currentiteration, failingorbder);
        }else{
            printall("fcartesian", normal,  currentiteration, failinggrid);
            printall("flfcartesian", normal,  currentiteration, failinglyapunov);
            printall("flfpcartesian", normal,  currentiteration, failingorbder);
        }
    }
    
}


void LYAPUNOV::getnewalpha(int currentiteration, RBFMETHOD &rbf)
{
    double summing=0.0;
    double normalizationfactor=0.0;
    rbf.alphavector.resize(rbf.alphasize);
    for(int iii=0; iii<rbf.alphasize; ++iii)
    {
        summing=0.0;
        for(int j=0; j<2*points_directional;++j)
        {
            summing+=orbder((2*points_directional)*(iii)+j);
        }
        if(summing>0.0)
        {
            summing=0.0;
        }
        rbf.alphavector(iii)=summing/((double)(2*points_directional));
        normalizationfactor+=rbf.alphavector(iii);
    }
    rbf.alphavector=abs(rbf.alphasize/normalizationfactor)*rbf.alphavector;
    if(printing)
    {
        printall("alphavector", normal,  currentiteration, rbf.alphavector);
    }
    
}

void LYAPUNOV::make_lyap_direcional(WENDLAND &wendland, RBFMETHOD &rbf)
{
    
    int i;
    wall_clock timer1;
    
    for(i=0; i<totaliterations; i++)
    {
        iteration(i);
        timer1.tic();
        
        
        lyapequation(i, rbf);
        lyapunovfunctions(i, true, rbf.directgrid, wendland, rbf);
        getnewalpha(i, rbf);
        chainrecurrentset(i, true, true, rbf.directgrid);
        firstderivative(i, true, rbf.directgrid, wendland, rbf);
        normed.clear();
        secondderivative(rbf.directgrid, wendland, rbf);
        findingeigenamount(i, true, rbf);
        *outputf  << "The whole proceedure for iteration no. " << i << " lasted: " << timer1.toc() << " sec"<< endl;
    }
    
}


void LYAPUNOV::make_lyap_direc_and_cart(double cart_grid_density, WENDLAND &wendland, RBFMETHOD &rbf)
{
    
    int i;
    wall_clock timer1;
    
    mat localcar;
    
    rbf.evaluatinggrid(ode_dimension, cart_grid_density, localcar);
    rbf.effectivegridnewnew(localcar);
    localcar.clear();
    
    bool decide;
    for(i=0; i<totaliterations; i++)
    {
        iteration(i);
        *outputf  << " " << endl;
        *outputf  << " " << endl;
        *outputf  << "Directional grid:" << endl;
        *outputf  << " " << endl;
        *outputf  << " " << endl;
        timer1.tic();
        lyapequation(i, rbf);
        decide=true;
        lyapunovfunctions(i,decide, rbf.directgrid, wendland, rbf);
        chainrecurrentset(i, decide, true,  rbf.directgrid);
        firstderivative(i, decide, rbf.directgrid, wendland, rbf);
        normed.clear();
        secondderivative(rbf.directgrid, wendland, rbf);
        findingeigenamount(i, decide, rbf);
        getnewalpha(i, rbf);
        
        failinggrid.clear();
        failinglyapunov.clear();
        failingorbder.clear();
        direcnozero.clear();
        negcollocation.clear();
        crslyapun.clear();
        crsorbder.clear();
        lyapfunc.clear();
        orbder.clear();
        fdvector.clear();
        sdvector.clear();
        TOTALEIGEN.clear();
        TOTALEIGENV.clear();
        
        decide=false;
        *outputf  << " " << endl;
        *outputf  << " " << endl;
        *outputf  << "Cartesian grid:" << endl;
        *outputf  << " " << endl;
        *outputf  << " " << endl;

        lyapunovfunctions(i,decide, rbf.cartesianevalgrid, wendland, rbf);
        chainrecurrentset(i, decide, true, rbf.cartesianevalgrid);
        firstderivative(i, decide, rbf.cartesianevalgrid, wendland, rbf);
        normed.clear();
        secondderivative(rbf.cartesianevalgrid, wendland, rbf);
        findingeigenamount(i, decide, rbf);
        
        failinggrid.clear();
        failinglyapunov.clear();
        failingorbder.clear();
        direcnozero.clear();
        negcollocation.clear();
        crslyapun.clear();
        crsorbder.clear();
        lyapfunc.clear();
        orbder.clear();
        fdvector.clear();
        sdvector.clear();
        TOTALEIGEN.clear();
        TOTALEIGENV.clear();
        
        *outputf  << "The whole proceedure for iteration no. " << i << " lasted: " << timer1.toc() << " sec"<< endl;
    }
    
}


void LYAPUNOV::make_chainrecurrent_eigenvalues(double cart_grid_density, WENDLAND &wendland, RBFMETHOD &rbf)
{
    
    int i;
    wall_clock timer1;
    
    mat localcar;
    rbf.evaluatinggrid(ode_dimension, cart_grid_density, localcar);
    rbf.effectivegridnewnew(localcar);
    localcar.clear();
    bool decide;
    for(i=0; i<totaliterations; i++)
    {
        iteration(i);
        *outputf  << " " << endl;
        *outputf  << " " << endl;
        *outputf  << "Directional grid:" << endl;
        *outputf  << " " << endl;
        *outputf  << " " << endl;

        timer1.tic();
        lyapequation(i, rbf);
        decide=true;
        lyapunovfunctions(i,decide, rbf.directgrid, wendland, rbf);
        chainrecurrentset(i, decide, true, rbf.directgrid);
        firstderivative(i, decide, failinggrid, wendland, rbf);
        normed.clear();
        secondderivative(failinggrid, wendland, rbf);
        findingeigenamount(i, decide, rbf);
        getnewalpha(i, rbf);

        failinggrid.clear();
        failinglyapunov.clear();
        failingorbder.clear();
        direcnozero.clear();
        negcollocation.clear();
        crslyapun.clear();
        crsorbder.clear();
        lyapfunc.clear();
        orbder.clear();
        fdvector.clear();
        sdvector.clear();
        TOTALEIGEN.clear();
        TOTALEIGENV.clear();
        
        decide=false;
        *outputf  << " " << endl;
        *outputf  << " " << endl;
        *outputf  << "Cartesian grid:" << endl;
        *outputf  << " " << endl;
        *outputf  << " " << endl;

        lyapunovfunctions(i,decide, rbf.cartesianevalgrid, wendland, rbf);
        chainrecurrentset(i, decide, true, rbf.cartesianevalgrid);
        firstderivative(i, decide, failinggrid, wendland, rbf);
        normed.clear();
        secondderivative(failinggrid, wendland, rbf);
        findingeigenamount(i, decide, rbf);

        failinggrid.clear();
        failinglyapunov.clear();
        failingorbder.clear();
        direcnozero.clear();
        negcollocation.clear();
        crslyapun.clear();
        crsorbder.clear();
        lyapfunc.clear();
        orbder.clear();
        fdvector.clear();
        sdvector.clear();
        TOTALEIGEN.clear();
        TOTALEIGENV.clear();
        
        *outputf  << "The whole proceedure for iteration no. " << i << " lasted: " << timer1.toc() << " sec"<< endl;
    }
    
}



void LYAPUNOV::make_norm_chain_recurrent_sets(double cart_grid_density, WENDLAND &wendland, RBFMETHOD &rbf)
{
    
    int i;
    wall_clock timer1;
    if(critval<0)
    {
        *outputf << "EXECUTION STOPPED " << endl;
        *outputf << "The critical value must be positive or zero, you are chosing to obtain the chain-recurrent set with the norm and the norm is positive or zero " << endl;
        finalization();
        exit(0);
    }
    mat localcar;
    rbf.evaluatinggrid(ode_dimension, cart_grid_density, localcar);
    rbf.effectivegridnewnew(localcar);
    localcar.clear();
    bool decide;
    for(i=0; i<totaliterations; i++)
    {
        iteration(i);
        *outputf  << " " << endl;
        *outputf  << " " << endl;
        *outputf  << "Directional grid:" << endl;
        *outputf  << " " << endl;
        *outputf  << " " << endl;

        timer1.tic();
        lyapequation(i, rbf);
        decide=true;
        lyapunovfunctions(i, decide, rbf.directgrid, wendland, rbf);
        firstderivative(i, decide, rbf.directgrid, wendland, rbf);
        chainrecurrentset(i, decide, false, rbf.directgrid);
        secondderivative(failinggrid, wendland, rbf);
        findingeigenamount(i, decide, rbf);
        getnewalpha(i, rbf);
        failinggrid.clear();
        failinglyapunov.clear();
        failingorbder.clear();
        direcnozero.clear();
        negcollocation.clear();
        crslyapun.clear();
        crsorbder.clear();
        lyapfunc.clear();
        orbder.clear();
        fdvector.clear();
        sdvector.clear();
        normed.clear();
        TOTALEIGEN.clear();
        TOTALEIGENV.clear();
        
        decide=false;
        *outputf  << " " << endl;
        *outputf  << " " << endl;
        *outputf  << "Cartesian grid:" << endl;
        *outputf  << " " << endl;
        *outputf  << " " << endl;

        lyapunovfunctions(i,decide, rbf.cartesianevalgrid, wendland, rbf);
        firstderivative(i, decide, rbf.cartesianevalgrid, wendland, rbf);
        chainrecurrentset(i, decide, false, rbf.cartesianevalgrid);
        secondderivative(failinggrid, wendland, rbf);
        findingeigenamount(i, decide, rbf);
        failinggrid.clear();
        failinglyapunov.clear();
        failingorbder.clear();
        direcnozero.clear();
        negcollocation.clear();
        crslyapun.clear();
        crsorbder.clear();
        lyapfunc.clear();
        orbder.clear();
        fdvector.clear();
        sdvector.clear();
        normed.clear();
        TOTALEIGEN.clear();
        TOTALEIGENV.clear();
        
        *outputf  << "The whole proceedure for iteration no. " << i << " lasted: " << timer1.toc() << " sec"<< endl;
    }
    
}
