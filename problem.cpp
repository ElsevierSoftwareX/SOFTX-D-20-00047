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
#include <fstream>
#include <armadillo>
#include <list>
#include "odesystem.hpp"
#include "instructions.hpp"
#include "wendland.hpp"
#include "RBF.hpp"
#include "generalities.hpp"
#include "lyapunovfunction.hpp"
#include "odetools.hpp"

using namespace std;
using namespace arma;

unsigned long long int glovar::functionodecalls=0;
std::ofstream glovar::outputf;
arma::span const All=span::all;


int main()
{
    system("pwd");
    wall_clock timer,timer1;
    timer.tic();
    glovar::outputf.open("output.lpx", fstream::out);
    printinformation();
    
    if(glovar::eigenvaluesjudge)
    {
        mat criticalpoints;
        criticalpoints.set_size(1,1);
        if((int)criticalpoints.n_cols!=glovar::ode_dimension)
        {
            glovar::outputf<<"The program has stopped because your critical values' matrix has a different dimension to: " << glovar::ode_dimension << endl;
        }
        criticalpoints<<0.0 << endr;
        crit_point_eigen_pairs(criticalpoints);
    }
    
    
    WENDLAND wendland(glovar::l,glovar::k,glovar::c,glovar::outputf);
    
    RBFMETHOD rbf(glovar::alpha, glovar::points_directional, glovar::radius, glovar::ode_dimension, glovar::min_geometric_limits, glovar::max_geometric_limits, glovar::normal, glovar::printing, glovar::outputf);
    
    LYAPUNOV lyapunov(glovar::totaliterations, glovar::ode_dimension, glovar::cart_grid_scaling, glovar::l, glovar::k, glovar::c, glovar::points_directional, glovar::critval, glovar::normal, glovar::printing, glovar::outputf);
    
    printhour(0);
    
    rbf.makeRBF(wendland);
    
    switch (glovar::computation_type){
        case glovar::choose_the_calculation::only_directional:
            lyapunov.make_lyap_direcional(wendland, rbf);
            break;
        case glovar::choose_the_calculation::directional_and_cartesian:
            lyapunov.make_lyap_direc_and_cart(glovar::cart_grid_scaling, wendland, rbf);
            break;
        case glovar::choose_the_calculation::chain_recurrent_set_eigenvalues:
            lyapunov.make_chainrecurrent_eigenvalues(glovar::cart_grid_scaling, wendland, rbf);
            break;
        case glovar::choose_the_calculation::norm_chain_recurrent_set:
            lyapunov.make_norm_chain_recurrent_sets(glovar::cart_grid_scaling, wendland, rbf);
            break;
    }
    
    finalization();  
    glovar::outputf  << "The whole proceedure last: " << timer.toc() << " sec"<< endl;
    printhour(1);
    glovar::outputf.close();
    
    return 0;
    
}




