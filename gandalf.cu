#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cufft.h>
#include <time.h>
#include <netcdf.h>
#include <fenv.h>
#include <assert.h>

#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(2);};
#define DEBUGPRINT(_fmt, ...)  if (debug) fprintf(stderr, "[file %s, line %d]: " _fmt, __FILE__, __LINE__, ##__VA_ARGS__)

#define CP_ON_GPU(to, from, isize) cudaMemcpy(to, from, isize, cudaMemcpyDeviceToDevice)
#define CP_TO_GPU(gpu, cpu, isize) cudaMemcpy(gpu, cpu, isize, cudaMemcpyHostToDevice)
#define CP_TO_CPU(cpu, gpu, isize) cudaMemcpy(cpu, gpu, isize, cudaMemcpyDeviceToHost)

#define CUDA_DEBUG(_fmt, ...)  if (debug) fprintf(stderr, "[file %s, line %d]: " _fmt, __FILE__, __LINE__, ##__VA_ARGS__, cudaGetErrorString(cudaGetLastError()))

#define ADDSUBT(a1, a2, a3, a4) addsubt <<<dG,dB>>> (a1, a2, a3, a4)
#define LINSTEP(a1, a2, a3) linstep <<<dG,dB>>> (a1, a2, a3)
#define GRADIENT(f, dfdx, dfdy) deriv <<<dG,dB>>> (f, dfdx, dfdy)


/////////////////////////////
// Input parameters
/////////////////////////////
//Device
int devid;

// algorithm choices
bool debug, restart;
bool linonly, nlrun;
int nwrite, nforce; // Set nforce very large if driven is false
int nsteps;  // nsteps defaults to zero
float maxdt, cfl;

// Initial conditions
bool decaying, driven, orszag_tang, noise;
float aw_coll;

//Computation Grid
__constant__ int Nx, Ny, Nz, zThreads;
__constant__ size_t Nkc, Nkf;
__constant__ float X0, Y0, Z0;

float endtime;

// Grid, block setup for GPU
dim3 dG, dB;
int totalThreads;

// forcing
int nkstir;
int *kstir_x, *kstir_y, *kstir_z;
int gm_nkstir;
int *gm_kstir_x, *gm_kstir_y, *gm_kstir_z;
float fampl, gm_fampl;

// Alfvenic packet (moving to the right)
int kpeak;

// dissipation
int alpha_z, alpha_hyper;
float nu_kz, nu_hyper;

// Internal variables

char *runname;
char *restartname;
char stopfile[255];
bool file_exists(char *filename);
void read_namelist(char *filename);
void restartRead(cuComplex* zp, cuComplex* zm, float* tim);
void restartWrite(cuComplex* zp, cuComplex* zm, float tim);
FILE *energyfile, *alf_kzkpfile, *awp_collfile, *awm_collfile;

//Dummy arrays
cuComplex *temp1, *temp2, *temp3;
cuComplex *padded;
cuComplex *dx, *dy;
float *fdxR, *fdyR, *gdxR, *gdyR;

cufftHandle plan_C2R, plan_R2C, plan2d_C2R;

// NetCDF info

struct NetCDF_ids {

    int file;
    int gpu;
    int restart;
    int kz_dim, kpar_dim, kperp_dim, t_dim, stir_dim;
    int kz, kpar, kperp, t, nsteps;
    int b2, v2;
    int nx, ny, nz;
    int x0, y0, z0;
    int alpha_z, nu_kz, alpha_hyper, nu_hyper;
    int nkstir, fampl;
    int kstir_x, kstir_y, kstir_z;
    int nwrite, nforce, maxdt, restart_name, cfl;
    int kpeak;

    int kperps[2];
    int  kpars[2];
    int kstir[1];

    int kparperp[3];

    size_t start_1d[2];
};

struct NetCDF_ids id;

//////////////////////////////////////////////////////////////////////
// Include files
//////////////////////////////////////////////////////////////////////
#include "c_fortran_namelist3.c"
#include "kernels.cu"
#include "maxReduc.cu"
#include "diagnostics.cu"
#include "nonlin.cu"
#include "courant.cu"
#include "forcing.cu"
#include "timestep.cu"

// Declare all required arrays
// Declare host variables
cuComplex *f, *g;

// Declare device variables
cuComplex *f_d, *g_d;
cuComplex *f_d_tmp, *g_d_tmp;


// Declare functions
void allocate_arrays();
void destroy_arrays();
void setup_device();
void setup_grid_block();
void finit(cuComplex *f, cuComplex *g);

//////////////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
    if ( argc < 1 ) {
        printf( "Usage: ./gandalf runname");
    } else {
        feenableexcept(FE_INVALID | FE_OVERFLOW);

        // Assuming argv[1] is the runname
        runname = argv[1];
        char str[255];
        strcpy(str, runname);
        strcat(str, ".in");
        printf("Reading from %s \n", str);
        // Read namelist
        read_namelist(str);

        *&Nkc = Nx*(Ny/2+1)*Nz*sizeof(cuComplex);
        *&Nkf = Nx*(Ny/2+1)*Nz*sizeof(float);

        cudaMemcpyToSymbol(Nkc, &Nkc, sizeof(size_t));
        cudaMemcpyToSymbol(Nkf, &Nkf, sizeof(size_t));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Allocate arrays
        allocate_arrays();

        // Setup dimgrid and dimblock
        setup_grid_block();

        // Stopfile
        strcpy(stopfile, runname);
        strcat(stopfile, ".stop");

        // Initialize diagnostics
        id = init_netcdf_diag(id);
        init_diag();

        // Create FFT plans
        fft_plan_create();

        //////////////////////////////////////////////////////////
        //    float dt = 1.e-2;
        float dt = maxdt;
        float tim=0;
        int istep=0;
        srand( (unsigned) time(NULL));
        //////////////////////////////////////////////////////////
        //If not restarting, initialize

        if(!restart){
            DEBUGPRINT("Not restarting. \n");

            //////////////////////////////////////////////////////////
            // Initialize Phi and Psi
            finit(f, g);

            // Transfer the fields to Device
            CP_TO_GPU(f_d, f, Nkc);
            CUDA_DEBUG("f Initialization on device: %s\n");

            CP_TO_GPU(g_d, g, Nkc);
            CUDA_DEBUG("g Initialization on device: %s\n");

        }

        // If restarting:
        else{

            DEBUGPRINT("Restarting. \n");
            restartRead(f_d, g_d, &tim);

            printf("Time after restart: %f \n", tim);
        }


        // Zeroth step
        if (nlrun) courant(&dt, f_d, g_d);
        DEBUGPRINT("dt = %f\n", dt);
        advance(f_d_tmp, f_d, g_d_tmp, g_d, dt, istep);
        diagnostics(f_d, g_d, tim, 0, id);
        istep++;
        tim+=dt;
        while(istep < nsteps) {

            if(istep % nwrite == 0) {
                DEBUGPRINT("t= %g \t dt= %g \t istep= %i\n",tim,dt,istep);
                diagnostics(f_d, g_d, tim, istep/nwrite, id);
                CUDA_DEBUG("CUDA check-in: %s\n");	
            }

            // Check CFL condition
            if (nlrun) courant(&dt, f_d, g_d);

            // Time advance RMHD & slow mode equations
            advance(f_d_tmp, f_d, g_d_tmp, g_d, dt, istep);

            tim+=dt;
            istep++;

            // Check if stopfile exists.
            if(file_exists(stopfile)) {
                printf("Simulation stopped by user with stopfile \n");
                break;
            }
            fflush(stdout);
        }
        if ((nsteps>0) && (istep % nwrite==0)) diagnostics(f_d, g_d, tim, istep/nwrite, id);

        DEBUGPRINT("Before restart write \n");
        restartWrite(f_d, g_d, tim);
        DEBUGPRINT("Restart write done \n");
        printf("Done.\n");


        float elapsed_time;
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("Total time(ms) = %f \n", elapsed_time);
        printf("Average time (ms) = %f \n", elapsed_time/istep);

        ////////////////////
        // Clean up Area
        ////////////////////

        // Destroy fft plan and close diagnostics
        fft_plan_destroy();
        close_netcdf_diag(id);

        // Free all arrays
        destroy_arrays();
        if(driven) {free(kstir_x); free(kstir_y); free(kstir_z);}

    }
    return 0;
}


bool file_exists(char *filename){
    if(FILE *file = fopen(filename, "r")) {
        fclose(file);
        return true;
    }
    return false;
}

void read_namelist(char* filename) {
    struct fnr_struct nml = fnr_read_namelist_file(filename);

    int debug_i, restart_i;
    int linonly_i;
    int decaying_i, driven_i, orszag_tang_i, noise_i;

    //Device
    if (fnr_get_int(&nml, "dev", "devid", &devid)) devid = 0;
    setup_device();
    // algo

    if (fnr_get_int(&nml, "algo", "debug", &debug_i)) debug_i=1;
    if(debug_i == 0) { debug = false; } else {debug = true;}

    if (fnr_get_int(&nml, "algo", "linonly", &linonly_i)) linonly_i=1;
    if (linonly_i == 0) { linonly = false; } else {linonly = true;}

    nlrun = true;
    if (linonly) {nlrun=false;}

    if( nlrun )
        printf("Performing nonlinear run.\n");
    else
        printf("Performing linear run.\n");

    if (fnr_get_int(&nml, "algo", "restart", &restart_i)) restart_i = 0;
    if (restart_i == 0) { restart = false;} else {restart = true;}

    restartname="restart";
    if(restart) fnr_get_string(&nml, "algo", "rest", &restartname);

    if (fnr_get_int(&nml,   "algo", "nwrite", &nwrite)) nwrite = 1;
    if (fnr_get_int(&nml,   "algo", "nforce", &nforce)) nforce = 1;
    if (fnr_get_float(&nml, "algo", "maxdt",  &maxdt))  maxdt = .1;
    if (fnr_get_float(&nml, "algo", "cfl",    &cfl))    cfl = .1;

    DEBUGPRINT("Algo read \n");

    // Initial conditions
    decaying = true;
    fnr_get_int(&nml, "init", "decaying", &decaying_i);
    if(decaying_i == 0) decaying = false;

    driven = true;
    fnr_get_int(&nml, "init", "driven", &driven_i);
    if(driven_i == 0) driven = false;

    orszag_tang = true;
    fnr_get_int(&nml, "init", "orszag_tang", &orszag_tang_i);
    if(orszag_tang_i == 0) orszag_tang = false;

    if(orszag_tang) printf("orszag_tang \n");

    noise = true;
    fnr_get_int(&nml, "init", "noise", &noise_i);
    if(noise_i == 0) noise = false;

    if (fnr_get_float(&nml, "init", "aw_coll", &aw_coll)) aw_coll = 0. ;

    if (fnr_get_int(&nml, "init", "kpeak", &kpeak)) kpeak=8;

    DEBUGPRINT("Initial conditions read \n");
    //Grid
    if (fnr_get_int(&nml, "grid", "Nx", &Nx)) *&Nx = 16;
    cudaMemcpyToSymbol(Nx, &Nx, sizeof(int));

    if (fnr_get_int(&nml, "grid", "Ny", &Ny)) *&Ny = 16;
    cudaMemcpyToSymbol(Ny, &Ny, sizeof(int));

    if (fnr_get_int(&nml, "grid", "Nz", &Nz)) *&Nz = 16;
    cudaMemcpyToSymbol(Nz, &Nz, sizeof(int));

    if (fnr_get_float(&nml, "grid", "X0", &X0)) *&X0 = 1.0f;
    cudaMemcpyToSymbol(X0, &X0, sizeof(float));

    if (fnr_get_float(&nml, "grid", "Y0", &Y0)) *&Y0 = 1.0f;
    cudaMemcpyToSymbol(Y0, &Y0, sizeof(float));

    if (fnr_get_float(&nml, "grid", "Z0", &Z0)) *&Z0 = 1.0f;
    cudaMemcpyToSymbol(Z0, &Z0, sizeof(float));

    if (fnr_get_int(&nml, "grid", "nsteps", &nsteps)) nsteps = 0;

    DEBUGPRINT("Grid read \n");

    // Dissipation
    if (fnr_get_int   (&nml, "dissipation", "alpha_z",     &alpha_z)) alpha_z = 2;
    if (fnr_get_float (&nml, "dissipation", "nu_kz",       &nu_kz)) nu_kz = 1.0f;
    if (fnr_get_int   (&nml, "dissipation", "alpha_hyper", &alpha_hyper)) alpha_hyper = 2;
    if (fnr_get_float (&nml, "dissipation", "nu_hyper",    &nu_hyper)) nu_hyper = 1.0f;

    DEBUGPRINT("Dissipation read \n");

    // Forcing
    fnr_get_int   (&nml, "forcing", "nkstir", &nkstir);
    fnr_get_float (&nml, "forcing", "fampl",  &fampl);

    if(driven){

        kstir_x = (int*) malloc(sizeof(int)*nkstir);
        kstir_y = (int*) malloc(sizeof(int)*nkstir);
        kstir_z = (int*) malloc(sizeof(int)*nkstir);


        // Initialize Forcing modes
        char tmp_str[255];
        char buffer[20];
        int f_k;
        for(int ikstir=0; ikstir<nkstir; ikstir++) {

            strcpy(tmp_str,"stir_");
            sprintf(buffer,"%d", ikstir);
            strcat(tmp_str, buffer);
            fnr_get_int(&nml, tmp_str, "kx", &f_k);
            kstir_x[ikstir] = (f_k + Nx) % Nx;
            fnr_get_int(&nml, tmp_str, "ky", &f_k);
            kstir_y[ikstir] = (f_k + Ny) % Ny;
            fnr_get_int(&nml, tmp_str, "kz", &f_k);
            kstir_z[ikstir] = (f_k + Nz) % Nz;

        }
    }

    DEBUGPRINT("Forcing read \n");
}


//////////////////////////////////////////////////////////////////////
// Restart routines
//////////////////////////////////////////////////////////////////////

void restartWrite(cuComplex* zp, cuComplex* zm, float tim) {
    DEBUGPRINT("Entering restart write\n");
    char str[255];
    FILE *restart;
    strcpy(str, runname);
    strcat(str,".res");
    restart = fopen(str, "wb");
    DEBUGPRINT("Opened restart file to write\n");

    cuComplex *zp_h;
    cuComplex *zm_h;

    DEBUGPRINT("Declared arrays \n");
    DEBUGPRINT("Nx = %d, Ny = %d, Nz = %d \n", Nx, Ny, Nz);

    zp_h = (cuComplex*) malloc(Nkc);
    CP_TO_CPU(zp_h, zp, Nkc);
    CUDA_DEBUG("Copying over zp %s\n");

    zm_h = (cuComplex*) malloc(Nkc);
    CP_TO_CPU(zm_h, zm, Nkc);
    CUDA_DEBUG("Copying over zm %s\n");

    DEBUGPRINT("Allocated and filled arrays \n");

    fwrite (&tim, sizeof(float), 1, restart);
    DEBUGPRINT("Wrote istep, tim \n");

    fwrite(zp_h, Nkc, 1, restart);
    DEBUGPRINT("Wrote zp \n");

    fwrite(zm_h, Nkc, 1, restart);
    DEBUGPRINT("Wrote zm \n");

    fclose(restart);
    free(zp_h); free(zm_h);

}

void restartRead(cuComplex* zp, cuComplex* zm, float* tim) {
    char stri[255];
    FILE *restart;
    strcpy(stri, restartname);
    strcat(stri,".res");
    DEBUGPRINT("Restartfile = %s \n", stri);
    restart = fopen(stri, "rb");

    cuComplex *zp_h;
    cuComplex *zm_h;

    zp_h = (cuComplex*) malloc(Nkc);
    zm_h = (cuComplex*) malloc(Nkc);

    fread(tim,sizeof(float),1,restart);

    fread(zp_h, Nkc, 1, restart);
    CP_TO_GPU(zp, zp_h, Nkc);

    fread(zm_h, Nkc, 1, restart);
    CP_TO_GPU(zm, zm_h, Nkc);

    fclose(restart);

    free(zp_h);
    free(zm_h);
}


// Setup device
void setup_device(){

    // Device information
    int ct, dev;
    struct cudaDeviceProp prop;

    cudaGetDeviceCount(&ct);
    printf("Device Count: %d\n",ct);

    cudaSetDevice(devid);
    cudaGetDevice(&dev);
    printf("Device ID: %d\n",dev);

    cudaGetDeviceProperties(&prop,dev);
    printf("Device Name: %s\n", prop.name);

    printf("Major mode: %d\n", prop.major);
    printf("Global Memory (bytes): %lu\n", (unsigned long) prop.totalGlobalMem);
    printf("Shared Memory per Block (bytes): %lu\n", (unsigned long) prop.sharedMemPerBlock);
    printf("Registers per Block: %d\n", prop.regsPerBlock);
    printf("Warp Size (threads): %d\n", prop.warpSize);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Size of Block Dimension (threads): %d * %d * %d\n", prop.maxThreadsDim[0],
            prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max Size of Grid Dimension (blocks): %d * %d * %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

}
void setup_grid_block(){
    int dev;
    struct cudaDeviceProp prop;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop,dev);

    //////////////////////////////////////////////////////////
    //set up normal dimGrid/dimBlock config
    int zBlockThreads = prop.maxThreadsDim[2];
    *&zThreads = zBlockThreads*prop.maxGridSize[2];
    totalThreads = prop.maxThreadsPerBlock;

    if(Nz > zBlockThreads) dB.z = zBlockThreads;
    else dB.z = Nz;

    int xy = (int) totalThreads/dB.z;
    int blockxy = (int) sqrt((float) xy);

    //dB = threadsPerBlock, dG = numBlocks
    dB.x = blockxy;
    dB.y = blockxy;

    if(Nz>zThreads) {
        dB.x = (unsigned int) sqrt((float) totalThreads/zBlockThreads);
        dB.y = (unsigned int) sqrt((float) totalThreads/zBlockThreads);
        dB.z = zBlockThreads;
    }

    dG.x = (unsigned int) ceil((float) Nx/dB.x + 0);
    dG.y = (unsigned int) ceil((float) Ny/dB.y + 0);
    if(prop.maxGridSize[2]==1) dG.z = 1;
    else dG.z = (unsigned int) ceil((float) Nz/dB.z) ;
    cudaMemcpyToSymbol(zThreads, &zThreads, sizeof(int));
    printf("zthreads = %d, zblockthreads = %d \n", zThreads, zBlockThreads);

    printf("dimGrid = %d, %d, %d \t dimBlock = %d, %d, %d \n", dG.x, dG.y, dG.z, dB.x, dB.y, dB.z);
}

////////////////////////////////////////
// Array allocation/destruction functions
////////////////////////////////////////
// Allocate arrays
void allocate_arrays()
{

    printf("Allocating arrays...\n");
    // Allocate host arrays
    f = (cuComplex*) malloc(Nkc);
    g = (cuComplex*) malloc(Nkc);

    // Allocate device arrays
    cudaMalloc((void**) &f_d, Nkc);
    cudaMalloc((void**) &g_d, Nkc);
    cudaMalloc((void**) &f_d_tmp, Nkc);
    cudaMalloc((void**) &g_d_tmp, Nkc);

    cudaMalloc((void**) &temp1, Nkc);
    cudaMalloc((void**) &temp2, Nkc);
    cudaMalloc((void**) &temp3, Nkc);

    cudaMalloc((void**) &fdxR, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &fdyR, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &gdxR, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &gdyR, sizeof(float)*Nx*Ny*Nz);

    cudaMalloc((void**) &dx, Nkc);
    cudaMalloc((void**) &dy, Nkc);

    cudaMalloc((void**) &padded, sizeof(cuComplex)*Nx*Ny*Nz);

}
void destroy_arrays(){

    // Destroy host arrays
    free(f); free(g);

    // Destroy device arrays containing fields
    cudaFree(f_d); cudaFree(g_d);

    // Destroy dummy arrays
    // Fields
    cudaFree(f_d_tmp); cudaFree(g_d_tmp);

    // nonlin
    cudaFree(temp1); cudaFree(temp2); cudaFree(temp3);

    cudaFree(fdxR); cudaFree(fdyR); cudaFree(gdxR); cudaFree(gdyR);

    cudaFree(dx); cudaFree(dy);

    // courant
    cudaFree(padded);
}

//////////////////////////////////////////////////////////////////////
// Alfven initialization
//////////////////////////////////////////////////////////////////////
void finit(cuComplex *f, cuComplex *g)
{
    int iky, ikx, ikz, index;
    float ran;

    // Zero out f and g
    for(ikz = 0; ikz<Nz; ikz ++){
        for(ikx = 0; ikx < Nx; ikx++){
            for(iky = 0; iky < Ny/2+1; iky++){
                index = iky + (Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;

                f[index].x = 0.0;
                f[index].y = 0.0;

                g[index].x = 0.0;
                g[index].y = 0.0;

            }
        }
    }

    ////////////////////////////////////////////////////////////////////////
    //	// Random initialization
    ////////////////////////////////////////////////////////////////////////
    if(noise){
        float k2;
        float ampl = 1.e+0/(Nx*Ny*Nz);
        for(iky=1;iky<=(Ny-1)/3+1; iky++){
            for(ikz=1;ikz<=(Nz-1)/3+1; ikz++){
                for(ikx=1;ikx<=(Nx-1)/3+1; ikx++){

                    index = iky + (Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;
                    //k2 = pow((float) iky/Y0,2) + pow((float) ikx/X0,2) + pow((float) ikz/Z0,2);
                    k2 = pow((float) iky/Y0,2)  + pow((float) ikz/Z0,2);

                    // AVK: working Density of states effect?
                    ran = ((float) rand()) / ((float) RAND_MAX + 1);
                    f[index].x = (sqrt(ampl/k2)/k2) * cos(ran*2.0*M_PI);
                    f[index].y = (sqrt(ampl/k2)/k2) * sin(ran*2.0*M_PI);
                    ran = ((float) rand()) / ((float) RAND_MAX + 1);
                    g[index].x = (sqrt(ampl/k2)/k2) * cos(ran*2.0*M_PI);
                    g[index].y = (sqrt(ampl/k2)/k2) * sin(ran*2.0*M_PI);

                    /*index = + (Ny/2+1)*(Nx-ikx) + (Ny/2+1)*Nx*(Nz-ikz);

                    // AVK: working Density of states effect?
                    ran = ((float) rand()) / ((float) RAND_MAX + 1);
                    f[index].x = sqrt(ampl) * cos(ran*2.0*M_PI);
                    f[index].y = sqrt(ampl) * sin(ran*2.0*M_PI);
                    ran = ((float) rand()) / ((float) RAND_MAX + 1);
                    g[index].x = sqrt(ampl) * cos(ran*2.0*M_PI);
                    g[index].y = sqrt(ampl) * sin(ran*2.0*M_PI);*/


                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////
    //	// Decaying Alfven cascade run
    ////////////////////////////////////////////////////////////////////////
    if(decaying){
        //float xi0 = 1.e+2/((Nx/3)*(Ny/3)*(Nz/3));
        float xi0 = 1.e-2;

        for(ikz=1; ikz<Nz/4; ikz++){
            for(ikx=1; ikx<(Nx-1)/3; ikx++){
                for(iky=1; iky<(Ny-1)/3 ; iky++){

                    index = iky + (Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;

                    ran = ((float) rand()) / ((float) RAND_MAX + 1);

                    f[index].x = cos(ran * 2.0* M_PI) * sqrt(xi0 * pow(iky,-10.0f/1.0f));
                    f[index].y = sin(ran * 2.0* M_PI) * sqrt(xi0 * pow(iky,-10.0f/1.0f));

                    //ran = ((float) rand()) / ((float) RAND_MAX + 1);
                    g[index].x = cos(ran * 2.0* M_PI) * sqrt(xi0 * pow(iky,-10.0f/1.0f));
                    g[index].y = sin(ran * 2.0* M_PI) * sqrt(xi0 * pow(iky,-10.0f/1.0f));
                }
            }
        }

        for(ikz = 3*Nz/4; ikz<Nz; ikz++){
            for(ikx=2*Nx/3 + 1; ikx<Nx; ikx++){
                for(iky=1; iky<Ny/3; iky++){

                    index = iky + (Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;

                    ran = ((float) rand()) / ((float) RAND_MAX + 1);

                    f[index].x = cos(ran * 2.0* M_PI) * sqrt(xi0 * pow(iky,-10.0f/1.0f));
                    f[index].y = sin(ran * 2.0* M_PI) * sqrt(xi0 * pow(iky,-10.0f/1.0f));

                    //ran = ((float) rand()) / ((float) RAND_MAX + 1);
                    g[index].x = cos(ran * 2.0* M_PI)* sqrt(xi0 * pow(iky,-10.0f/1.0f));
                    g[index].y = sin(ran * 2.0* M_PI) * sqrt(xi0 * pow(iky,-10.0f/1.0f));
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////
    //	// Orszag-Tang initial conditions
    ////////////////////////////////////////////////////////////////////////
    if(orszag_tang){
        //phi = -2(cosx + cosy)
        //A = 2cosy + cos2x
        //f = z+ = phi + A
        //g = z- = phi - A

        ikx = 1; iky = 0; ikz = 0;
        index = iky + (Ny/2+1) * ikx + (Ny/2+1)*Nx*ikz;

        f[index].x = -1.0;
        g[index].x = -1.0;

        ikx = Nx-1; iky = 0; ikz = 0;
        index = iky + (Ny/2+1) * ikx + (Ny/2+1)*Nx*ikz;

        f[index].x = -1.0;
        g[index].x = -1.0;

        ikx = 2; iky = 0; ikz = 0;
        index = iky + (Ny/2+1) * ikx + (Ny/2+1)*Nx*ikz;

        f[index].x = 0.50;
        g[index].x = -0.50;

        ikx = Nx-2; iky = 0; ikz = 0;
        index = iky + (Ny/2+1) * ikx + (Ny/2+1)*Nx*ikz;

        f[index].x = 0.50;
        g[index].x = -0.50;


        ikx = 0; iky = 1; ikz = 0;
        index = iky + (Ny/2+1) * ikx + (Ny/2+1)*Nx*ikz;

        g[index].x = -2.0;
    }


    ////////////////////////////////////////////////////////////////////////
    //	// Alfven wave collisions -- GGH
    ////////////////////////////////////////////////////////////////////////

    // f = z+ = sin(x - z)
    // g = z- = -sin(y + z)

    ikx = 1; iky=0; ikz = Nz-1;
    index = iky + (Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;

    //f[index].y = -aw_coll * 0.5;

    ikx = Nx-1; iky=0; ikz = 1;
    index = iky + (Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;

    //f[index].y = aw_coll * 0.5;
    float aw_envelope;
    ikx=0; iky=1;
    for (ikz = 0 ; ikz<Nz/2+1 ; ikz++)
    {
        aw_envelope = 0.5 * aw_coll * exp(-0.05*(pow(ikz-kpeak, 2)));
        //   aw_envelope = 0.5 * aw_coll * exp(-0.5*(pow(ikz-kpeak, 2)));
        index = iky+(Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;
        //   printf ("ikz = %i,\t aw = %g \n", ikz, aw_envelope);
        g[index].x = aw_envelope;
    }

}
