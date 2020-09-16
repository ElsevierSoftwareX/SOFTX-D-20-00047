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

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <iterator>
#include <armadillo>
#include <algorithm>
#include <iomanip>      // std::setprecision
#include "wendland.hpp"
#include "generalities.hpp"
#if defined(_OPENMP)
#include <omp.h>
#endif


using namespace std;
using namespace arma;
arma::span const All=span::all;

WENDLAND::WENDLAND(int l,int k, double c, ofstream &outputf){
    this->l=l;
    this->k=k;
    this->c=c;
    this->outputf=&outputf;
}

void WENDLAND::wendlandfunction()
{
    //NOTE: comodin is a vector that inside the loop works the divisors of the fractions that the integration produces: Then computes the LCD. After the loop is finished. Then it works the numerators of such fractions and computes the MCD.
    ///wdlf(0,All) IS THE EXPONENTS
    ///wdnf(1,All) IS THE COEFFICIENTS
    ///wdnf(2,All) IS THE EXPONENT OF C-smooth parameter
    wall_clock timer;
    timer.tic();
    int N=l+1;
    int finaldim=N+2*k;
    Mat <double> comodin(1,(int)(finaldim));
    wdlfunction.resize(3,finaldim);
    wdlfunction.zeros();
    comodin.zeros();
    comodin.fill(1.0);
    
    
    long mcm=0;
    long mcd=0;
    ///FIRST LET US COMPUTE (1-r)^l. NOTICE that it is being computing in the entire vector, so it will include several zeros.
    /// THE HIGHEST EXPONENT STARTS WITH THE VECTOR
    wdlfunction.resize(3,(int)(N));
    double factormcm=0.0;
    
    pascal(wdlfunction);
    
    int counter=N;
    for(int j=0; j<k; ++j)
    {
        factormcm=0.0;
        /// PRODUCT BY t
        tprod(wdlfunction);
        ++counter;
        wdlfunction.resize(3,(int)(counter));
        fixindex(wdlfunction);
        
        ///// Integrating
        tprod(wdlfunction);
        ++counter;
        wdlfunction.resize(3,(int)(counter));
        comodin.resize(1,(int)(counter));
        fixindex(wdlfunction);
        fixindex(comodin);
        wdlfunction.resize(3,(int)counter);
        wdlfunction(0,0)=0;
        comodin(0,0)=1;
        for(int i=counter-1; i>=1; --i)
        {
            wdlfunction(1,i)=wdlfunction(1,i)/wdlfunction(0,i);
            comodin(0,i)=(i)*comodin(0,i-1);
            factormcm+=wdlfunction(1,i);
        }
        factormcm+=wdlfunction(0,0);
        wdlfunction(1,All)*=-1.0;
        wdlfunction(1,0)=factormcm;
        Row<double> temp((int)counter);
        temp=comodin(0,All);
        mcm=getmcm(temp);
        comodin(0)=mcm;
    }
    
    //NOW: comodin works the numerators
    comodin.resize((int)(counter));
    comodin.fill(1.0);
    comodin=round(mcm*wdlfunction(1,All));
    Row<double> temp((int)counter);//
    temp=comodin(0,All);
    mcd=getmcd(temp);
    
    wdlfunction(1,All)=round((mcm/mcd)*wdlfunction(1,All));
    wdlfunction(2,All)=wdlfunction(0,All);
    
    
    *outputf  << "Computing the Wendland Function lasted: " << timer.toc() << " sec"<< endl;
    
}

void WENDLAND::wendlandderivative(mat &wdlfinput, mat &auxfunc)
{
    int dim=(int)wdlfinput.n_cols;
    auxfunc.resize(3,dim);
    auxfunc.zeros();
    for(int i=0; i<dim; ++i)
    {
        auxfunc(0,i)=i-2;
        auxfunc(1,i)=wdlfinput(0,i)*wdlfinput(1,i);
    }
    auxfunc(2,All)=wdlfinput(2,All);
    cleanzeros(auxfunc);
}


double WENDLAND::evawdlfn(double r, mat const &wdlfinput)
{
    double wdlfvalue=0.0;
    double checking=0.0;
    wdlfvalue=0.0;
    
    int dim = (int)wdlfinput.n_cols;
    {
        checking=1.0-c*r;
        if(checking>0.0)
        {
            for(int i=0; i<dim; ++i)
            {
                wdlfvalue+=wdlfinput(1,i)*pow(r,wdlfinput(0,i))*pow(c,wdlfinput(2,i));
            }
        }else{
            wdlfvalue=0.0;
        }
    }
    return wdlfvalue;   
}





void WENDLAND::pascal(mat &vector1) const
{
    //    int negcounter=0;
    for(int i=0;i<=l;++i)
    {
        vector1(0,i)=i;
        double x=1.0;
        for(int h=0;h<=i;h++)
        {
            //negcounter=i-h;
            vector1(1,h)=pow(-1,h)*x;
            x = x * (i - h) / (h + 1.0);
        }
    }
}

void WENDLAND::tprod(mat &vector)
{
    int locallength=(int)vector.n_cols;
    vector(0,All)+=ones(locallength).t();
    vector(2,All)+=ones(locallength).t();
}

void WENDLAND::fixindex(mat &vector)
{
    int locallength=(int)vector.n_cols;
    int localwidth=(int)vector.n_rows;
    Mat<double> conmutevec;
    if(localwidth>1)
    {
        conmutevec.zeros(localwidth,locallength);
        for(int i=1; i<locallength; ++i)
        {
            conmutevec(0,i)=vector(0,i-1);
            conmutevec(1,i)=vector(1,i-1);
            conmutevec(2,i)=vector(2,i-1);
        }
        vector=conmutevec;
    }else{
        conmutevec.zeros(localwidth,locallength);
        for(int i=1; i<locallength; ++i)
        {
            conmutevec(0,i)=vector(0,i-1);
        }
        vector=conmutevec;
    }
}



char WENDLAND::signaturing(double unknownsign)
{
    char algebraicsymbol;
    
    if(arma::sign(unknownsign)>=0)
    {
        algebraicsymbol='+';
    }else{
        algebraicsymbol='-';
    }
    return algebraicsymbol;
}

void WENDLAND::cleanzeros(mat &vector)
{
    int localdimension=(int)vector.n_cols;
    Mat<double> temp(3,localdimension);
    temp.zeros();
    int count=0;
    for(int i=0; i<localdimension; ++i)
    {
        if(vector(1,i)==0)
        {
            ++count;
        }else{
            break;
        }
    }
    temp.resize(3,localdimension-count);
    temp(All,All)=vector(All,span(count,localdimension-1));
    vector.resize(3,localdimension-count);
    vector=temp;
}


void WENDLAND::mcmp(long long unsigned value1, long long unsigned value2, long long unsigned &valueout)
{
    while (value2 > 0) {
        long long unsigned r = value1 % value2;
        value1 = value2;
        value2 = r;
    }
    valueout=value1;
}

void WENDLAND::gcmp(long long unsigned value1, long long unsigned value2, long long unsigned &valueout)
{
    while (value2 > 0) {
        long long unsigned r = value1 % value2;
        value1 = value2;
        value2 = r;
    }
    valueout=value1;
}

long WENDLAND::getmcm(arma::mat &matrix)
{
    long long unsigned valueout=1;
    long long unsigned mcm=matrix(0,0);
    matrix=abs(matrix);
    for(int i=0; i<=(int)matrix.n_cols-1; ++i)
    {
        if((mcm == 0) || (matrix(0,i) == 0)){
            break;
        }else{
            mcmp(mcm, matrix(0,i),valueout);
            mcm=(mcm * matrix(0,i)) / valueout; //
        }
    }
    return mcm;
}

long long unsigned WENDLAND::getmcd(mat &vector)
{
    long long unsigned valueout=1;
    long long unsigned mcd=vector(0,0);
    vector=abs(vector);
    for(int i=1; i<(int)vector.n_cols;++i)
    {
        if(vector(0,i)!=0.0)
        {
            gcmp(mcd, abs(vector(0,i)),valueout);
            mcd=valueout;
        }
    }
    return mcd;
}


