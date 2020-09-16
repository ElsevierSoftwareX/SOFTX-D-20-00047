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
#include "generalities.hpp"
#include "odesystem.hpp"
#include <iomanip>

using namespace arma;
using namespace std;

arma::span const All=span::all;

void printinformation()
{
    ofstream datos;
    
    datos.open ("data.lpx", fstream::app);
    datos << "Required data needed to reproduce this computation of a CLF." << endl;
    datos << "The instruction given to perform the computations is: " << glovar::probnames[glovar::computation_type] << endl;
    
        datos << "The hexagonal grid is used";
    
    
    if(glovar::normal)
    {
        datos << " under the normalising method approach ";
    }else{
        datos << " under the unnormalised method approach ";
    }
    if(glovar::totaliterations>0)
    {
        datos << "for a total of " << glovar::totaliterations << " iterations." << endl;
    }else{
        datos << "for only one single computation of a Lyapunov function." << endl;
    }
    
    datos << "The general settings for this problem are: " << endl;
    datos <<  " WENDLAND Function Parameters:"  << " l=" << glovar::l << ", k=" << glovar::k << ", c=" << glovar::c << " || " << endl;
    datos << "alpha: " << glovar::alpha << " | " << endl;
    datos << "The minima geometric limits are: " << endl;
    for(int jc=0; jc<glovar::ode_dimension; ++jc)
    {
        datos << glovar::min_geometric_limits[jc] << " | ";
    }
    datos << endl;
    
    datos << "The maxima geometric limits are: " << endl;
    for(int jc=0; jc<glovar::ode_dimension; ++jc)
    {
        datos << glovar::max_geometric_limits[jc] << " | ";
    }
    datos << endl;
    
    datos << "For the iterative grid: " << endl;
    datos << "the radius is: " << glovar::radius << " for the total amount of points: " << glovar::points_directional << " | " << " the critical value gamma is: " << glovar::critval << " || " << endl;
    datos << "For the cartesian grid, the scaling parameter is: " << glovar::cart_grid_scaling << endl;
    datos << " " << endl;
    datos.close();
}







void printall(const string nombre, bool normal, int ordernum, mat &vectoraimprimmir)
{
    wall_clock timer;
    timer.tic();
    const std::string output_file_extension="m";
    int dim=(int)vectoraimprimmir.n_rows;
    int dim2=(int)vectoraimprimmir.n_cols;
    if((dim!=0)&&(dim2!=0))
    {
        glovar::outputf << "Printing results..." << endl;
        if(dim2 >= dim)
        {
            for(int totalwidth=0; totalwidth<dim; ++totalwidth)
            {
                ostringstream fileName;
                if(normal)
                {
                    fileName<<"s"<<nombre<<"var"<<totalwidth<<"dm."<<output_file_extension;
                }else{
                    fileName<<"s"<<nombre<<"var"<<totalwidth<<"."<<output_file_extension;
                }
                ofstream valuesdocument(fileName.str(), fstream::out | fstream::app);
                if(normal)
                {
                    valuesdocument << nombre<< "var"<<totalwidth << "ite" << ordernum << "=[";
                }else{
                    valuesdocument << nombre<< "var"<<totalwidth << "ite" << ordernum << "dm=[";
                }
                for(int p=0; p<dim2; ++p)
                {
                    valuesdocument << std::fixed << std::setprecision(18) << vectoraimprimmir(totalwidth,p) << " " ;
                }
                valuesdocument << "];" << endl;
                valuesdocument.close();
            }
        }else{
            for(int totalwidth=0; totalwidth<dim2; ++totalwidth)
            {
                ostringstream fileName;
                if(normal)
                {
                    fileName<<"s"<<nombre<<"var"<<totalwidth<<"dm."<<output_file_extension;
                }else{
                    fileName<<"s"<<nombre<<"var"<<totalwidth<<"."<<output_file_extension;
                }
                ofstream valuesdocument(fileName.str(), fstream::out | fstream::app);
                if(normal)
                {
                    valuesdocument << nombre<< "var"<<totalwidth << "ite" << ordernum << "=[";
                }else{
                    valuesdocument << nombre<< "var"<<totalwidth << "ite" << ordernum << "dm=[";
                }
                for(int p=0; p<dim; ++p)
                {
                    valuesdocument << std::fixed << std::setprecision(18) << vectoraimprimmir(p,totalwidth) << " " ;
                }
                valuesdocument << "];" << endl;
                valuesdocument.close();
            }
        }

        glovar::outputf  << "The whole proceedure to print results lasted: " << timer.toc() << " sec"<< endl;
    }else{
        glovar::outputf << "WARNING: " << nombre << " does not contain values and its dimension is 0 " << endl;
    }
    printhour(1);
}

void printcube(const string nombre, bool normal, int ordernum, cube &vectoraimprimmir)
{
    wall_clock timer;
    timer.tic();
    const std::string output_file_extension="m";  
    int dim=(int)vectoraimprimmir.n_rows;
    int dim2=(int)vectoraimprimmir.n_cols;
    int dim3=(int)vectoraimprimmir.n_slices;
    if((dim!=0)&&(dim2!=0))
    {
        glovar::outputf << "Printing results..." << endl;
        {
            for(int totalwidth=0; totalwidth<dim; ++totalwidth)
            {
                ostringstream fileName;
                if(normal)
                {
                    fileName<<"s"<<nombre<<"var"<<totalwidth<<"dm."<<output_file_extension;
                }else{
                    fileName<<"s"<<nombre<<"var"<<totalwidth<<"."<<output_file_extension;
                }
                ofstream valuesdocument(fileName.str(), fstream::out | fstream::app);
                if(normal)
                {
                    valuesdocument << nombre<< "var"<<totalwidth << "ite" << ordernum << "=[";
                }else{
                    valuesdocument << nombre<< "var"<<totalwidth << "ite" << ordernum << "dm=[";
                }
                for(int nsl=0; nsl<dim3; ++nsl)
                {
                    for(int p=0; p<dim2; ++p)
                    {
                        valuesdocument << std::fixed << std::setprecision(18) << vectoraimprimmir(totalwidth,p,nsl) << " " ;
                    }
                }
                valuesdocument << "];" << endl;
                valuesdocument.close();
            }
        }
        glovar::outputf  << "The whole proceedure to print results lasted: " << timer.toc() << " sec"<< endl;
    }else{
        glovar::outputf << "WARNING: " << nombre << " does not contain values and its dimension is 0 " << endl;
        finalization();
        exit(0);
    }
    printhour(1);
}

void printhour(const int &definition)
{
    time_t tempus;
    struct tm * infotiempo;
    char datos[100];
    
    time (&tempus);
    infotiempo = localtime(&tempus);
    if(definition==0)
    {
        strftime(datos,sizeof(datos),"Computation started on %d-%m-%Y at %H:%M:%S",infotiempo);
        
    }if(definition==1)
    {
        strftime(datos,sizeof(datos),"Computation finished on %d-%m-%Y at %H:%M:%S",infotiempo);
    }
    string str(datos);
    
    glovar::outputf  << str << endl;
    
}

void iteration(int i)
{
    glovar::outputf  << "================================= Iteration no. " << i << " =================================" << endl;
}

void finalization()
{
    glovar::outputf  << " " << endl;
    glovar::outputf  << "================================= FINALIZATION =================================" << endl;
    glovar::outputf  << " " << endl;
    glovar::outputf  << "No. of times the function was called: " << glovar::functionodecalls << endl;
}
