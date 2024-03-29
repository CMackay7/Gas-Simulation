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
#include <mpi.h>       // header for MPI


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

  struct timeval timer[4];

  /* vars for MPI */
  int numProcesses, rankNum;

  double start=omp_get_wtime();
  int count = 0;
  /* input size of system */
  num = atoi(argv[1]);
  timesteps = atoi(argv[2]);
  eps = atof(argv[3]);
  //printf("Initializing for %d particles in x,y,z space... \n", num);

  //printf("with wall boundaries at x,y +/-%f\n", BOXWALL);
  //printf("and eps=%g as determining molecules at same point\n", eps);

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
    //printf("\n ERROR in malloc for (at least) vz - aborting\n");
    return -99;
  }
  else {
    //printf("  (malloc-ed)  ");
  }
  rc = init(mass, x, y, z, vx, vy, vz, num);
  if (rc != 0) {
   // printf("\n ERROR during init() - aborting\n");
    return -99;
  }
  else {
   // printf("  INIT COMPLETE\n");
  }
  totalMass = 0.0;
  for (i=0; i<num; i++) {
    
    totalMass += mass[i];
  }
  meanEnergy = calc_mean_energy(totalMass, vx, vy, vz, num);
  //printf("Time zero. Mean energy=%g\n", meanEnergy);


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


  //printf("Now to integrate for %d timesteps\n", timesteps);



  MPI_Init(NULL, NULL);

  gettimeofday(&timer[0], NULL);
  //MPI_timer[0] = MPI_Wtime();


  MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
  MPI_Comm_rank(MPI_COMM_WORLD, &rankNum);
  int moleculesPerProcess = round(num/numProcesses);
  //int moleculesPerProcess = reinterpret_cast<int>( moleculesPerPrp );

  //printf("molecules per process %d\n", moleculesPerProcess);
  float x_buffer[ moleculesPerProcess ];
  float y_buffer[ moleculesPerProcess ];
  float z_buffer[ moleculesPerProcess ];
  float vx_buffer[ moleculesPerProcess ];
  float vy_buffer[ moleculesPerProcess ];
  float vz_buffer[ moleculesPerProcess ];

  // time=0 was initial conditions
  // There is no cross fetching so this whole loop can be done in parrallel
  //printf("about to start running the loop \n");
  for (time=1; time<=timesteps; time++) {
    int counter = 0;
    for(i=(rankNum * moleculesPerProcess); i<((rankNum * moleculesPerProcess) + moleculesPerProcess -1); i++) {
      //remove nested loop and replace all i for ranknum
      // calc potential new position
      
      new_x = x[i] + vx[i];
      new_y = y[i] + vy[i];
      new_z = z[i] + vz[i];
      //printf("checking against walls");
      // check if wall boundary crossed in which case bounce (i.e. reflection) with 10% loss of momentum
      if (new_x < -BOXWALL || new_x > +BOXWALL) {
        x_buffer[counter] = x[i] - vx[i];
        vx_buffer[counter] = -vx[i]*0.9;
      }
      else {
        x_buffer[counter] = new_x;
        vx_buffer[counter] = vx[i];
      }

      if (new_y < -BOXWALL || new_y > +BOXWALL) {
        y_buffer[counter] = y[i] - vy[i];
        vy_buffer[counter] = -vy[i]*0.9;
      }
      else {
        y_buffer[counter] = new_y;
        vy_buffer[counter] = vy[i];
      }

      if (new_z < -BOXWALL || new_z > +BOXWALL) {
        z_buffer[counter] = z[i] - vz[i];
        vz_buffer[counter] = -vz[i]*0.9;
      }
      else {
        z_buffer[counter] = new_z;
        vz_buffer[counter] = vz[i];
      }
      counter = counter + 1;
      // if(time == 2) {
      //   printf("%.6f,", vx[i]);
      // }
      
    }

      //Order is always x y z
    float newValuesToSend[3] = {new_x, new_y, new_z};
    int tag = 999;
    //printf("Done checking againts walls");
    float *x_rbuf;
    float *y_rbuf;
    float *z_rbuf;
    float *vx_rbuf;
    float *vy_rbuf;
    float *vz_rbuf;



    //printf("started passing stuff");
      if (rankNum>0) {
        //printf("the problem is with non rank 0 gather rank=%d \n", rankNum);
        MPI_Gather(&x_buffer, moleculesPerProcess, MPI_FLOAT, &x_rbuf, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(&y_buffer, moleculesPerProcess, MPI_FLOAT, NULL, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(&z_buffer, moleculesPerProcess, MPI_FLOAT, NULL, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(&vx_buffer, moleculesPerProcess, MPI_FLOAT, NULL, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(&vx_buffer, moleculesPerProcess, MPI_FLOAT, NULL, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(&vx_buffer, moleculesPerProcess, MPI_FLOAT, NULL, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);

        MPI_Bcast(x, num, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(y, num, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(z, num, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(vx, num, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(vy, num, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(vx, num, MPI_FLOAT, 0, MPI_COMM_WORLD);
      }  else {
        
        int tempi = 0;
        if (count == 0) {
          for(tempi; tempi < moleculesPerProcess; tempi++){
            printf("%.6f,", vz_buffer[tempi]);
          }
        }
        count++;

        x_rbuf =  (float *) malloc(num * sizeof(float));
        y_rbuf =  (float *) malloc(num * sizeof(float));
        z_rbuf =  (float *) malloc(num * sizeof(float));
        vx_rbuf = (float *) malloc(num * sizeof(float));
        vy_rbuf = (float *) malloc(num * sizeof(float));
        vz_rbuf = (float *) malloc(num * sizeof(float));
        // Gathering from each process
        MPI_Gather(&x_buffer, moleculesPerProcess, MPI_FLOAT, x_rbuf, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(&y_buffer, moleculesPerProcess, MPI_FLOAT, y_rbuf, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(&z_buffer, moleculesPerProcess, MPI_FLOAT, z_rbuf, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(&vx_buffer, moleculesPerProcess, MPI_FLOAT, vx_rbuf, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(&vx_buffer, moleculesPerProcess, MPI_FLOAT, vy_rbuf, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(&vx_buffer, moleculesPerProcess, MPI_FLOAT, vz_rbuf, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
        // Gathering done will now broadcast to every process

        x = x_rbuf;
        y = y_rbuf;
        z = z_rbuf;
        vx = vx_rbuf;
        vy = vy_rbuf;
        vz = vz_rbuf;
        
        int tempcount = 0;
        
        

        MPI_Bcast(x_rbuf, num, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(y_rbuf, num, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(z_rbuf, num, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(vx_rbuf, num, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(vy_rbuf, num, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(vx_rbuf, num, MPI_FLOAT, 0, MPI_COMM_WORLD);

      }





      //MPI_Finalize();

      //if (rankNum > 0) {
       // printf("got to here_non 0 \n");
      //} else {
        //printf("got to_here 0 \n");
      //}





      //some sort of wait for all the threads to reach this point broadcast position

      // now check if new position is same as any other molecule's old postion in which case bounce all velocity components
      // given we are dealing with real numbers we need to say if 2 molecules within a given "eps" that they are at the same point

      // This might be able to be parrelised while it uses different mollecules it might be okay
      // But will have to check if a particle is in its old state or has bounced and then is colliding against new state

      int currMolecule = 0;
        int notcurrMolecule  = 0;
        for(notcurrMolecule = 0; notcurrMolecule < num; notcurrMolecule ++){
        for (currMolecule=(rankNum * moleculesPerProcess); currMolecule<((rankNum * moleculesPerProcess) + moleculesPerProcess -1); currMolecule++) {
          if (notcurrMolecule != currMolecule) {
            if (fabs(x[currMolecule]-x[notcurrMolecule])<eps && fabs(y[currMolecule]-y[notcurrMolecule])<eps && fabs(z[currMolecule]-z[notcurrMolecule])<eps) {
              // printf("i:%d (%f, %f, %f) hits j:%d (%f, %f, %f)\n", i,j,x[i],y[i],z[i],x[j],y[j],z[j]);
              // printf("x diff: %f, y diff: %f, z diff: %f; eps=%f\n",fabs(x[j]-x[i]), fabs(y[j]-y[i]), fabs(z[j]-z[i]), eps);
              //printf("time %d, molecule %d collides with %d\n",time,i,currMolecule);
              vx[currMolecule] = -vx[currMolecule];
              vy[currMolecule] = -vy[currMolecule];
              vz[currMolecule] = -vz[currMolecule];
              
            }
          }
        }
      }

    if (rankNum>0) {
      //printf("the problem is with non rank 0 gather rank=%d \n", rankNum);
      MPI_Gather(&x_buffer, moleculesPerProcess, MPI_FLOAT, &x_rbuf, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Gather(&y_buffer, moleculesPerProcess, MPI_FLOAT, NULL, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Gather(&z_buffer, moleculesPerProcess, MPI_FLOAT, NULL, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Gather(&vx_buffer, moleculesPerProcess, MPI_FLOAT, NULL, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Gather(&vx_buffer, moleculesPerProcess, MPI_FLOAT, NULL, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Gather(&vx_buffer, moleculesPerProcess, MPI_FLOAT, NULL, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }  else {
      x_rbuf =  (float *) malloc(num * sizeof(float));
      y_rbuf =  (float *) malloc(num * sizeof(float));
      z_rbuf =  (float *) malloc(num * sizeof(float));
      vx_rbuf = (float *) malloc(num * sizeof(float));
      vy_rbuf = (float *) malloc(num * sizeof(float));
      vz_rbuf = (float *) malloc(num * sizeof(float));
      // Gathering from each process
      MPI_Gather(&x_buffer, moleculesPerProcess, MPI_FLOAT, x_rbuf, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Gather(&y_buffer, moleculesPerProcess, MPI_FLOAT, y_rbuf, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Gather(&z_buffer, moleculesPerProcess, MPI_FLOAT, z_rbuf, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Gather(&vx_buffer, moleculesPerProcess, MPI_FLOAT, vx_rbuf, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Gather(&vx_buffer, moleculesPerProcess, MPI_FLOAT, vy_rbuf, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Gather(&vx_buffer, moleculesPerProcess, MPI_FLOAT, vz_rbuf, moleculesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
      // Gathering done will now broadcast to every process

      x = x_rbuf;
      y = y_rbuf;
      z = z_rbuf;
      vx = vx_rbuf;
      vy = vy_rbuf;
      vz = vz_rbuf;
    }
    
    } // gas molecules


    MPI_Finalize();
    /*
      DEBUG: output position of a given molecule (e.g. number 0)
      output_position_molecule(x, y, z, 0);  // output only given molecule
    */

   if (rankNum == 0) {
    meanEnergy = calc_mean_energy(totalMass, vx, vy, vz, num);
    printf("Time %d. Mean energy=%g\n", time, meanEnergy);
    //time steps

    meanEnergy = calc_mean_energy(totalMass, vx, vy, vz, num);
    printf("Time %d. Mean energy=%g\n", time, meanEnergy);

    printf("Time to init+solve %d molecules for %d timesteps is %g seconds\n", num, timesteps, omp_get_wtime()-start);
  }
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
