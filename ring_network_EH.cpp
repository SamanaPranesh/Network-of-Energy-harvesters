// Author : Samana Pranesh
// Affiliation : PhD Scholar, Department of Applied Mechanics & Biomedical Engineering
// Contact information : psamana97@gmail.com

// C++ code with OpenMP parallelization to simulate a ring network of electromagnetic pendulum energy harvesters.
// Algorithm used : 4th Order Runge-Kutta time marching algorithm

/*
   PARAMETERS DEFINED IN THE CODE :
   freq --> Amplitude of base excitation
   gamma_m and gamma_e --> damping parameters
   beta --> coupling strength
   N --> Number of energy harvesters in the network
   m --> degree of each node in the network

   For compiling: g++ test.cpp -o test -lm -lgsl -lgslcblas -Ofast -fopenmp

*/


#include<math.h>
#include<ctime>
#include<stdio.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_randist.h>
#include "network_lib.cpp"
#include<omp.h>

// Parameters are defined
#define freq 0.05
#define gamma_f 0.03
#define gamma_e 0.01
#define beta 0.1

// RHS of the ODE to be solved is defined in functions 'f' and 'g'
double f(double y[], int i)
{
    return y[i];
}

double g(double t, double x[], double y[], double omega, int i, int N, int **adj)
{
    double coupling=0;
    for(int j=0;j<N;j++)
    {
        coupling=coupling+(adj[i][j]*(sin(x[i])-sin(x[j]))*cos(x[i]) );
    }
    
    return freq*sin(omega*t) - (gamma_f+gamma_e)*y[i] - sin(x[i]) - beta*coupling ;
}


// Runge-Kutta algorithm with OpenMP parallelisation
void stepForward(double x[], double y[], double omega, int N, double dt, double t, int **adj)
{
    double val;
    double temp_x[N];
    double temp_y[N];
    double K[4*N];
    double L[4*N];
    
    #pragma omp parallel for
    for(int j=0;j<N;j++)
    {
        temp_x[j] = x[j];
        temp_y[j] = y[j];
    }

    for(int k=0;k<4;k++)
    {
        #pragma omp parallel for
        for(int i=0;i<N;i++)
        {
            
            K[k*N+i] = dt*f(temp_y,i);  // xdot = f(x)
            L[k*N+i] = dt*g(t,temp_x,temp_y,omega,i,N,adj); // ydot = g(x)
        }

        if(k<=2)
        {
            val=(k<=1)?0.5:1;
            
            #pragma omp parallel for
            for(int z=0;z<N;z++)
            {
                temp_x[z] = x[z] + val*K[k*N+z];
                temp_y[z] = y[z] + val*L[k*N+z];
            }
        }
     }
     
     #pragma omp parallel for
     for(int z=0;z<N;z++)
     {
        x[z] = x[z] + (1/6.0)*(K[z] + K[3*N+z] + (2*(K[N+z] + K[2*N+z]))); // always write 1/6.0 and not 1/6
        y[z] = y[z] + (1/6.0)*(L[z] + L[3*N+z] + (2*(L[N+z] + L[2*N+z])));
     }

}

// Initial Conditions
void init(double x[], double y[], int N, gsl_rng* h)
{
    #pragma omp parallel for
    for(int i=0;i<N;i++)
    {
        x[i] = gsl_ran_flat(h,0,1);
        y[i] = gsl_ran_flat(h,0,1);
    }
}


int main()
{
    gsl_rng *h=gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(h,time(NULL));
    srand48(time(NULL)); 

    int N=100;
    int m = 2;
    double dt = 0.01, t=0, omega;

    // Initialize adjacency matrix
    int **adj = new int*[N];
    for (int i = 0; i < N; i++) 
        {
            adj[i] = new int[N];
            for (int j = 0; j < N; j++) {
                adj[i][j] = 0; // Initialize all elements to 0
            }
        }

        // Connect each node to its adjacent nodes
        for (int i = 0; i < N; i++) 
        {
            int left_neighbor = (i - 1 + N) % N; // Node on the left
            int right_neighbor = (i + 1) % N;    // Node on the right
        
        // Establish connections
        adj[i][left_neighbor] = 1;
        adj[left_neighbor][i] = 1; // Connection in the opposite direction
        adj[i][right_neighbor] = 1;
        adj[right_neighbor][i] = 1; // Connection in the opposite direction
        }

    char name1[50]; 

    double x[N], y[N];
    double max[N];
    init(x,y,N,h);

            
    sprintf(name1,"N100_%dm_.dat",m);
    FILE *fp1=fopen(name1,"w+");
    
    omega = 0.8;

    while(omega<=1.15)
    {
        double t1=omp_get_wtime();
        for(int n=0;n<=1000000;n++)
        {
            stepForward(x,y,omega,N,dt,t,adj);
            if(n==700000)
            {
                for(int i=0;i<N;i++)
                {
                    max[i]=y[i];
                }
            }
            else if(n>700000)
            {
                for(int i=0;i<N;i++)
                {
                    if(y[i]>max[i])
                    max[i]=y[i];
                    
                } 
            }
            
            t = t+dt;

        }

        double power[N], total_power=0;

        for(int i=0;i<N;i++)
        {
            power[i] = gamma_e * max[i] *max[i];
            

        }

        for(int i=0;i<N;i++)
        {
            total_power = total_power + power[i];
        }

        double t2=omp_get_wtime();

        printf("%d %lf %lf %lf sec\n",m,omega,total_power/N,t2-t1);
        fprintf(fp1,"%d %lf %lf\n",m,omega,total_power/N);

        omega = omega+0.0071; 
    
    }


    for(int i=0;i<N;i++) 
    {
        delete[] adj[i];
    }
    delete[] adj;
    return 0;

}