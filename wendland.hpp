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

#ifndef wendland_hpp
#define wendland_hpp

#include <stdio.h>
#include <armadillo>
class WENDLAND{
public:
    WENDLAND(int l, int k, double c, std::ofstream &outputf);
    void wendlandfunction();
    
    void wendlandderivative(arma::mat &wdlfinput, arma::mat &auxfunc);
    
    double evawdlfn(double r, arma::mat const &wdlfn);
    
    void pascal(arma::mat &vector1) const;
    
    void tprod(arma::mat &cpower);
    
    void fixindex(arma::mat &vector);
    
    char signaturing(double unknownsign);
    
    void cleanzeros(arma::mat &vector);
    
    void mcmp(long long unsigned value1, long long unsigned value2, long long unsigned &valueout);
    
    void gcmp(long long unsigned value1, long long unsigned value2, long long unsigned &valueout);
    
    long getmcm(arma::mat &matrix);  
    
    long long unsigned getmcd(arma::mat &vector);
    
    
    double l;
    double k;
    double c;
    
    /*%%%% GENERALITIES %%%%*/
        
    std::ofstream* outputf;
    
    arma::mat wdlfunction;
    arma::mat wdlf1;
    arma::mat wdlf2;
    arma::mat wdlf3;
};

#endif /* wendland_hpp */
