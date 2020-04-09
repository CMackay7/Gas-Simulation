/*
  COMP328 - ASSIGNMENT 1
  (c) mkbane, University of Liverpool (2020)


  You are provided with this serial code.
  Your assignment is to parallelise, with MPI and/or OpenMP and/or to run on a GPU and to include any
  addition features required from the formal definition of your assignment. You should add all necessary
  error handling and your final parallel code should work, efficiently, for any number of parallel processing elements.

  This code is a simplistic model of gas molecules in a fixed box.
  Molecules bounce off any wall with a loss 10% of their momentum per collision.
  Molecules may bounce of another molecule but with no loss of momentum.
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define BOXWALL 10.0  // box width (-BOXWALL, +BOXWALL) & height (-BOXWALL, +BOXWALL)

int init(float*, float*, float*, float*, float*, float*, float*, int);
float calc_mean_energy(int, float*, float*, float*, int);
void output_position_molecule(float*, float*, float*, int);

int main(int argc, char* argv[]) {
  int i, j;
  int num;     // user defined (argv[1]) total number of gas molecules in simulation
  int time, timesteps; // for time stepping, including user defined (argv[2]) number of timesteps to integrate
  int rc;      // return code
  float *mass, *x, *y, *z, *vx, *vy, *vz;  // 1D array for mass, (x,y,z) position, (vx, vy, vz) velocity
  float totalMass, meanEnergy;  // for stats
  float new_x, new_y, new_z;    // potential new position for gas molecules
  float eps;                // user defined (argv[3]) for molecules to be in same position

  double start=omp_get_wtime();

  /* input size of system */
  num = atoi(argv[1]);
  timesteps = atoi(argv[2]);
  eps = atof(argv[3]);
  printf("Initializing for %d particles in x,y,z space...", num);
  printf("with wall boundaries at x,y +/-%f\n", BOXWALL);
  printf("and eps=%g as determining molecules at same point\n", eps);

  /* malloc arrays and pass ref to init(). NOTE: init() uses random numbers */
  mass = (float *) malloc(num * sizeof(float));
  x =  (float *) malloc(num * sizeof(float));
  y =  (float *) malloc(num * sizeof(float));
  z =  (float *) malloc(num * sizeof(float));
  vx = (float *) malloc(num * sizeof(float));
  vy = (float *) malloc(num * sizeof(float));
  vz = (float *) malloc(num * sizeof(float));
  // should check all rc but let's just see if last malloc worked
  if (vz == NULL) {
    printf("\n ERROR in malloc for (at least) vz - aborting\n");
    return -99;
  }
  else {
    printf("  (malloc-ed)  ");
  }
  rc = init(mass, x, y, z, vx, vy, vz, num);
  if (rc != 0) {
    printf("\n ERROR during init() - aborting\n");
    return -99;
  }
  else {
    printf("  INIT COMPLETE\n");
  }
  totalMass = 0.0;
  for (i=0; i<num; i++) {
    totalMass += mass[i];
  }
  meanEnergy = calc_mean_energy(totalMass, vx, vy, vz, num);
  printf("Time zero. Mean energy=%g\n", meanEnergy);


  /*
     MAIN TIME STEPPING LOOP
     in theory we can approximate movement by making delta t very small
     for now, we just normalise everything to "unit time" and approximate movements by
     (i) loop over each gas particle performing
         (a) if it hits a wall, it bounces back BUT loses 10% per velocity component

     given unit time, new position = old position + velocity UNLESS hits something


     We can APPROXIMATELY handle collisions with other particles by seeing if new position of molecule 'i'
     intersects with old position of molecule 'j'. We will only bounce our i molecule and NOT effect j molecule
     (to make this a tractable exercise!)
  */


  printf("Now to integrate for %d timesteps\n", timesteps);

  // time=0 was initial conditions
  for (time=1; time<=timesteps; time++) {
    for(i=0; i<num; i++) {
      // calc potential new position
      new_x = x[i] + vx[i];
      new_y = y[i] + vy[i];
      new_z = z[i] + vz[i];
      // check if wall boundary crossed in which case bounce (i.e. reflection) with 10% loss of momentum
      if (new_x < -BOXWALL || new_x > +BOXWALL) {
        x[i] = x[i] - vx[i];
        vx[i] = -vx[i]*0.9;
      }
      else {
        x[i] = new_x;
      }

      if (new_y < -BOXWALL || new_y > +BOXWALL) {
        y[i] = y[i] - vy[i];
        vy[i] = -vy[i]*0.9;
      }
      else {
        y[i] = new_y;
      }

      if (new_z < -BOXWALL || new_z > +BOXWALL) {
        z[i] = z[i] - vz[i];
        vz[i] = -vz[i]*0.9;
      }
      else {
        z[i] = new_z;
      }
      // now check if new position is same as any other molecule's old postion in which case bounce all velocity components
      // given we are dealing with real numbers we need to say if 2 molecules within a given "eps" that they are at the same point
      for (j=0; j<num; j++) {
        if (j != i) {
          if (fabs(x[j]-x[i])<eps && fabs(y[j]-y[i])<eps && fabs(z[j]-z[i])<eps) {
            // printf("i:%d (%f, %f, %f) hits j:%d (%f, %f, %f)\n", i,j,x[i],y[i],z[i],x[j],y[j],z[j]);
            // printf("x diff: %f, y diff: %f, z diff: %f; eps=%f\n",fabs(x[j]-x[i]), fabs(y[j]-y[i]), fabs(z[j]-z[i]), eps);
            printf("time %d, molecule %d collides with %d\n",time,i,j);
            vx[i] = -vx[i];
            vy[i] = -vy[i];
            vz[i] = -vz[i];
          }
        }
      }
    } // gas molecules

    /*
      DEBUG: output position of a given molecule (e.g. number 0)
      output_position_molecule(x, y, z, 0);  // output only given molecule
    */

    //meanEnergy = calc_mean_energy(totalMass, vx, vy, vz, num);
    //printf("Time %d. Mean energy=%g\n", time, meanEnergy);
  } // time steps

   meanEnergy = calc_mean_energy(totalMass, vx, vy, vz, num);
   printf("Time %d. Mean energy=%g\n", time, meanEnergy);

   printf("Time to init+solve %d molecules for %d timesteps is %g seconds\n", num, timesteps, omp_get_wtime()-start);

} // main


int init(float *mass, float *x, float *y, float *z, float *vx, float *vy, float *vz, int num) {
  /*
     use random numbers to set initial conditions
     NOTE: we have unrolled manually (one var per loop) to maximise vectorisation opportunity
  */
  int i;
  for (i=0; i<num; i++) {
    mass[i] = 0.001 + (float)rand()/(float)RAND_MAX;            // 0.001 to 1.001

    // particles start in middle volume
    float min=-BOXWALL/2, mult=BOXWALL;
    x[i] = min + mult*(float)rand()/(float)RAND_MAX;   //  -BOXWALL/2 to +BOXWALL/2 per axis
    y[i] = min + mult*(float)rand()/(float)RAND_MAX;   //
    z[i] = min + mult*(float)rand()/(float)RAND_MAX;   //

    vx[i] = -0.2 + 0.4*(float)rand()/(float)RAND_MAX;   // -0.2 to +0.2 per axis
    vy[i] = -0.2 + 0.4*(float)rand()/(float)RAND_MAX;
    vz[i] = -0.2 + 0.4*(float)rand()/(float)RAND_MAX;
  }

  return 0;
} // init


float calc_mean_energy(int mass, float *vx, float *vy, float *vz, int num) {
  /*
     energy is sum of 0.5*mass*velocity^2
     where velocity^2 is sum of squares of components
  */
  int i;
  float totalEnergy = 0.0, meanEnergy;
  for (i=0; i<num; i++) {
    totalEnergy += vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
  }
  totalEnergy = 0.5 * mass * totalEnergy;
  meanEnergy = totalEnergy / (float) num;
  return meanEnergy;
}

void output_position_molecule(float *x, float *y, float *z, int i) {
  printf("%d: %f, %f, %f\n", i, x[i], y[i], z[i]);
  return;
}
