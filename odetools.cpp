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


#include <armadillo>

#include "odesystem.hpp"
#include "instructions.hpp"
#include "generalities.hpp"
#include "odetools.hpp"

using namespace arma;
using namespace std;
arma::span const All=span::all;



void jacobian(const bool normal, rowvec const &x, mat &J, const double finitedifferencetol)
{
    wall_clock timer;
    timer.tic();
    
    const int N=(int)x.n_cols;
    J.set_size(N,N);
    
    rowvec xk(N), sk(N), f(N);
    
    odesystem(normal, x, f);
    
    for(int i=0; i< N; ++i)
    {
        xk=x;
        xk(i)+=finitedifferencetol;
        odesystem(normal,xk,sk);
        
        J(i,All)=(sk- f)/finitedifferencetol;
    }
    glovar::outputf  << "The whole proceedure to construct the Jacobian lasted: " << timer.toc() << " sec"<< endl;
    
    printhour(1);
    
}


void eigvalsol( const bool normal, rowvec const &x, cx_vec &eigval, cx_mat &eigvec, const double finitedifferencetol)
{
    wall_clock timer;
    timer.tic();
    
    const int N=(int)x.size();
    mat jacobianm(N,N);
    jacobian(normal,x,jacobianm,finitedifferencetol);
    eig_gen(eigval, eigvec, jacobianm);
    glovar::outputf  << "The whole proceedure to obtain the eigen-pairs lasted: " << timer.toc() << " sec"<< endl;
    
    printhour(1);
}




void judge(const bool normal, rowvec const &x, const double finitedifferencetol)
{
    wall_clock timer;
    timer.tic();
    
    mat judgematrix((int)x.size(),(int)x.size());
    
    cx_vec eigval;
    cx_mat eigvec;
    eigvalsol(normal, x, eigval, eigvec,finitedifferencetol);
    
    if(    ((real(eigval(0))!=0.0) && ((real(eigval(1))!=0.0)) && (imag(eigval(0))!=0.0) && (imag(eigval(1))!=0.0)))
    {
        if( (real(eigval(0)) < 0.0) && (real(eigval(1)) < 0.0) )
        {
            glovar::outputf  << "The critical point " << x << " is a Stable Focus  " << endl;
        }
        else if ( (real(eigval(0)) > 0.0) && (real(eigval(1)) > 0.0) )
        {
            glovar::outputf  << "The critical point " << x << " is a Unstable Focus  " << endl;
        }
    }
    
    if((imag(eigval(0))==0) && (imag(eigval(1))==0))
    {
        if(((real(eigval(0))<0) && (real(eigval(1))>0)) || ((real(eigval(0))>0) && (real(eigval(1))<0)))
        {
            glovar::outputf  << "The critical point " << x << " is a Saddle" << endl;
        }
        if( (real(eigval(0))<0) && (real(eigval(1))<0) )
        {
            glovar::outputf  << "The critical point " << x << " is a Stable Node" << endl;
        }
        if( (real(eigval(0))>0) && (real(eigval(1))>0) )
        {
            glovar::outputf  << "The critical point " << x << " is a Unstable Node" << endl;
        }
        
        
    }
    
    if( ((real(eigval(0))==0.0) && (imag(eigval(0))==0.0)) && ((imag(eigval(0))==0.0) && (imag(eigval(1))==0.0)) )
    {
        glovar::outputf  << "Linearization failed, something is wrong!! GAME OVER!! " << endl;
        
    }
    glovar::outputf  << "The whole proceedure to Judge the critical point lasted: " << timer.toc() << " sec"<< endl;
    
    printhour(1);
}

void crit_point_eigen_pairs(mat &matofpoints)
{
    double finitedifferencetol=1e-8;
    int dim=(int)matofpoints.n_rows;
    int dim2=(int)matofpoints.n_cols;
    rowvec inforvec(dim2);
    mat J(dim2,dim2);
    mat eigenvaluesm(dim2,dim2);
    cx_vec eigval;
    cx_mat eigvec;
    for(int i=0; i<dim; ++i)
    {
        inforvec=matofpoints(i,span());
        glovar::outputf  << "The point is: " << inforvec << endl;
        if(dim2==2)
        {
            jacobian(glovar::normal, inforvec, J, finitedifferencetol);
            glovar::outputf  << "The Jacobian is: " << J << endl;
            eigvalsol(glovar::normal, inforvec, eigval, eigvec, finitedifferencetol);
            glovar::outputf  << "The Eigenvalues are: " << eigenvaluesm << endl;
            judge(glovar::normal, inforvec, finitedifferencetol);
        }else{
            eigvalsol(glovar::normal, inforvec, eigval, eigvec, finitedifferencetol);
            int decideneg=0;
            int decidepos=0;
            for(int cr=0; cr<(int)eigval.n_rows; ++cr)
            {
                if(real(eigval(cr))<0.0)
                {
                    ++decideneg;
                }else{
                    ++decidepos;
                }
            }
            if(decideneg==(int)eigval.n_rows)
            {
                glovar::outputf << "The point " << inforvec << " is an attractor." << endl;
            }else if(decidepos==(int)eigval.n_rows){
                glovar::outputf << "The point " << inforvec << " is an repeller." << endl;
            }
        }
    }
}


