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

using namespace arma;


void odesystem(const bool normal, rowvec const &x, rowvec &f)
{
    f(0)=-1.0*x(0)*((x(0)*x(0))+(x(1)*x(1))-(1.0/4.0))*((x(0)*x(0))+(x(1)*x(1))-1.0)-x(1);
    f(1)=-1.0*x(1)*((x(0)*x(0))+(x(1)*x(1))-(1.0/4.0))*((x(0)*x(0))+(x(1)*x(1))-1.0)+x(0);
    


//DO NOT MODIFY BELOW THIS LINE.
    if(normal)
    {
        double norm2=0.0;
        double delta=10e-8;
        norm2=dot(f,f);
        f/=std::abs(sqrt(delta+norm2));
        
    }
    
    glovar::functionodecalls+=1;
}

