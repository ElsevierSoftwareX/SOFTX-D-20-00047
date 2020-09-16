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


#ifndef instructions_hpp
#define instructions_hpp


#include <stdio.h>
#include <iostream>
#include <armadillo>

namespace glovar {
extern unsigned long long int functionodecalls;
extern std::ofstream outputf;
enum choose_the_calculation {
    only_directional, directional_and_cartesian, chain_recurrent_set_eigenvalues, norm_chain_recurrent_set
};

char const probnames[][110]={"only_directional","directional_and_cartesian","chain_recurrent_set_eigenvalues", "norm_chain_recurrent_set"};

/* How do you want to solve your CLF?*/
const choose_the_calculation computation_type=directional_and_cartesian;

/*%%%% SECTION TO DEFINE PROBLEM TO ANALYZE %%%%*/

const int ode_dimension=2;

const double min_geometric_limits[ode_dimension]={-1.6,-1.6};

const double max_geometric_limits[ode_dimension]={1.6,1.6};

const double alpha=0.104;

/*%%%% SECTION TO DEFINE CONDITIONS %%%%*/

const bool normal=true;    /*true FOR THE ALMOST NORMALIZED METHOD*/

const bool eigenvaluesjudge=false;

const bool printing=true;

const double critval=-0.5;

const int points_directional=10; /* AMOUNT OF POINTS PER DIRECTION ON THE DIRECTIONAL GRID*/

const double radius=0.49;

/*%%%% SECTION TO DEFINE AMOUNT OF ITERATIONS %%%%*/

const int totaliterations=10;

/*%%%% SECTION TO DEFINE WENDLAND FUNCTION %%%%*/

const int l=5;
const int k=3;
const double c=1.0;

/* For the cartisian evaluation grid */  
const double cart_grid_scaling=0.20080;

const int OMP_NUM_THREADS=16;
};

#endif /* instructions_hpp */
