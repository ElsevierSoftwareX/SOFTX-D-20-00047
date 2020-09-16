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


#ifndef lyapunovfunction_hpp
#define lyapunovfunction_hpp
#include <stdio.h>
#include <armadillo>
#include <list>
#include "instructions.hpp"


class LYAPUNOV{
public:
    LYAPUNOV(int totaliterations, int ode_dimension, double cart_grid_density, int l, int k, double c, int points_directional, double critval, bool normal, bool printing, std::ofstream &outputf);
    
    void lyapequation(int currentiteration, RBFMETHOD &rbf);
    
    
    void lyapunovfunctions(int currentiteration, bool type_of_grid, arma::mat &evalcoordinates, WENDLAND &wendland, RBFMETHOD &rbf);
    
    void firstderivative(int currentiteration, bool type_of_grid, arma::mat &evalcoordinates, WENDLAND &wendland, RBFMETHOD &rbf);
    
    void secondderivative(arma::mat &evalcoordinates, WENDLAND &wendland, RBFMETHOD &rbf);
    
    void findingeigenamount(int currentiteration, bool type_of_grid, RBFMETHOD &rbf);
        
    void chainrecurrentset(int currentiteration, bool type_of_grid, bool with_orbder, arma::mat &evalcoordinates);
    
    void getnewalpha(int currentiteration, RBFMETHOD &rbf);
    
    void make_lyap_direcional(WENDLAND &wendland, RBFMETHOD &rbf);
    
    void make_lyap_dense(int currentiteration, WENDLAND &wendland, RBFMETHOD &rbf);
    
    void make_lyap_direc_and_cart(double cart_grid_density, WENDLAND &wendland, RBFMETHOD &rbf);
    
    void make_chainrecurrent_eigenvalues(double cart_grid_density, WENDLAND &wendland, RBFMETHOD &rbf);
    
    void make_lyap_all(double cart_grid_density, WENDLAND &wendland, RBFMETHOD &rbf);
    
    void make_norm_chain_recurrent_sets(double cart_grid_density, WENDLAND &wendland, RBFMETHOD &rbf);
    
    int totaliterations;
    int currentiteration;
    int ode_dimension;
    int l;
    int k;
    double c;
    int points_directional;
    double critval;
    double cart_grid_density;
    bool type_of_grid;
    bool normal;
    bool printing;
    double checking;
    std::ofstream* outputf;
    arma::rowvec lyapfunc;
    arma::rowvec orbder;
    arma::vec betaod;
    arma::mat fdvector;
    arma::cube sdvector;
    arma::rowvec normed;
    arma::mat TOTALEIGEN;
    arma::cube TOTALEIGENV;
    std::vector<bool> booldirectgrid;
    std::list<int> counter;
    std::list<int> counterncol;
    arma::mat failinggrid;
    arma::mat failinglyapunov;
    arma::mat failingorbder;
    arma::mat direcnozero;
    arma::mat negcollocation;
    arma::vec crslyapun;
    arma::vec crsorbder;
};
#endif /* lyapunovfunction_hpp */
