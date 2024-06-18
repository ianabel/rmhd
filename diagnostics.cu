//////////////////////////////////////////////////////////////////////
// Alfven diagnostics
//////////////////////////////////////////////////////////////////////

// kz- Kperp spectra of Alfven waves

void energy_kz_kperp(cuComplex* kPhi, cuComplex* kA, float time, int jstep, struct NetCDF_ids id)
{

    //  int ikpmax = (int) ceil( sqrt( pow((float)(Nx-1)/3, 2) + pow((float)(Ny-1)/3, 2) ) );
    //  float kpmax = ceil( sqrt( pow((float)(Nx-1)/3, 2) + pow((float)(Ny-1)/3, 2) ) );

    int ikpmax = (int) (Nx-1)/3;
    float kpmax = (float) (Nx-1)/3;
    float *kinEnergy_kp, *magEnergy_kp;

    float *totEnergy_h, *kinEnergy_h, *magEnergy_h;

    totEnergy_h = (float*) malloc(sizeof(float));
    kinEnergy_h = (float*) malloc(sizeof(float));
    magEnergy_h = (float*) malloc(sizeof(float));


    kinEnergy_h[0] = 0.;
    magEnergy_h[0] = 0.;

    cuComplex *kPhi_h;
    kPhi_h = (cuComplex*) malloc(Nkc);
    CP_TO_CPU(kPhi_h, kPhi, Nkc);
    int Nyc = (Ny/2+1);
    // want to pull kphi up here to the host and sum it
    for (int j=0; j<Ny/2+1; j++) {
        for (int i=0; i<Nx; i++) {
            for (int k=0; k<Nz; k++) {
                int idx = j+i*Nyc+k*Nx*Nyc;
                //	if(kPhi_h[idx].x>0.1) {printf("Element %d\t%d\t%d\t Value %g \n",i,j,k,kPhi_h[idx].x);}
                kinEnergy_h[0] = kinEnergy_h[0] + kPhi_h[idx].x;	
            }
        }
    }
    free(kPhi_h);
    printf("Kinetic Energy = %g \n", kinEnergy_h[0]);


    // Allocate arrays to hold kinetic and magnetic energy vs k on GPU
    cudaMalloc((void**) &kinEnergy_kp, sizeof(float)*ikpmax*Nz);
    cudaMalloc((void**) &magEnergy_kp, sizeof(float)*ikpmax*Nz);

    // Set array values to zero
    zero <<<Nz,ikpmax>>> (kinEnergy_kp, Nz*ikpmax,1,1);
    zero <<<Nz,ikpmax>>> (magEnergy_kp, Nz*ikpmax,1,1);

    //loop through the ky's
    for(int ikp=1; ikp<ikpmax; ikp++) {
        kz_kpshellsum <<<dG, dB>>> (kPhi, ikp, kinEnergy_kp);
        kz_kpshellsum <<<dG, dB>>> (kA, ikp, magEnergy_kp);
    }


    CUDA_DEBUG("kz_kpshellsum: %s\n");

    float *kinEnergy_kp_h, *magEnergy_kp_h;

    // Allocate arrays to hold kinetic and magnetic energy vs k on CPU
    kinEnergy_kp_h = (float*) malloc(sizeof(float)*ikpmax*Nz);
    magEnergy_kp_h = (float*) malloc(sizeof(float)*ikpmax*Nz);

    // Set array values to zero
    for (int i=0; i<ikpmax*Nz; i++) {
        kinEnergy_kp_h[i] = 0.;
        magEnergy_kp_h[i] = 0.;
    }

    CP_TO_CPU (kinEnergy_kp_h, kinEnergy_kp, sizeof(float)*ikpmax*Nz);
    CUDA_DEBUG("Copying shell sums KE: %s\n");
    CP_TO_CPU (magEnergy_kp_h, magEnergy_kp, sizeof(float)*ikpmax*Nz);
    CUDA_DEBUG("Copying shell sums ME: %s\n");


    size_t start[3], count[3];
    start[0] = jstep;
    start[1] = 0;
    start[2] = 0;

    count[0] = 1;
    count[1] = ikpmax-1;
    count[2] = Nz;

    int retval;

    if (retval = nc_put_vara(id.file, id.b2, start, count, magEnergy_kp_h)) ERR(retval);
    if (retval = nc_put_vara(id.file, id.v2, start, count, kinEnergy_kp_h)) ERR(retval);
    if (retval = nc_sync(id.file)) ERR(retval);

    kinEnergy_h[0] = 0.;
    magEnergy_h[0] = 0.;

    for (int i=0; i<ikpmax*Nz; i++) {
        kinEnergy_h[0] = kinEnergy_h[0] + kinEnergy_kp_h[i];
        magEnergy_h[0] = magEnergy_h[0] + magEnergy_kp_h[i];
    }
    totEnergy_h[0] = kinEnergy_h[0] + magEnergy_h[0];

    printf("Total Energy = %g\t Kin Energy = %g\t Magnetic Energy = %g\n", 
            totEnergy_h[0], kinEnergy_h[0], magEnergy_h[0]);

    free(totEnergy_h); free(kinEnergy_h); free(magEnergy_h); 
    cudaFree(kinEnergy_kp); cudaFree(magEnergy_kp);
    free(kinEnergy_kp_h); free(magEnergy_kp_h);

}

////////////////////////////////////////
// Total energy
void energy(cuComplex* kPhi, cuComplex* kA, float time, int jstep, struct NetCDF_ids id)
{
    DEBUGPRINT("Entering energy\n");

    cuComplex *padded;
    cudaMalloc((void**) &padded, sizeof(cuComplex)*Nx*Ny*Nz);

    cuComplex *totEnergy_h, *kinEnergy_h, *magEnergy_h;

    totEnergy_h = (cuComplex*) malloc(sizeof(cuComplex));
    kinEnergy_h = (cuComplex*) malloc(sizeof(cuComplex));
    magEnergy_h = (cuComplex*) malloc(sizeof(cuComplex));

    kinEnergy_h[0].x=0.;
    kinEnergy_h[0].y=0.;

    // integrate kA to find magnetic energy
    //sumReduc(magEnergy_h, kA, padded);
    sumReduc_gen(magEnergy_h, kA, padded, Nx, Ny, Nz);
    CUDA_DEBUG("sumreduc kA: %s\n");

    // integrate kPhi to find kinetic energy
    //sumReduc(kinEnergy_h, kPhi, padded);
    sumReduc_gen(kinEnergy_h, kPhi, padded, Nx, Ny, Nz);
    CUDA_DEBUG("sumreduc kPhi: %s\n");

    //calculate total energy
    totEnergy_h[0].x = kinEnergy_h[0].x + magEnergy_h[0].x;

    cudaFree(padded);

    printf("Total Energy = %g\t Kin Energy = %g\t Magnetic Energy = %g\n", 
            totEnergy_h[0].x, kinEnergy_h[0].x, magEnergy_h[0].x);

    size_t start[1],count[1];
    start[0] = jstep;
    count[0] = 1;
    int retval;
    if (retval = nc_put_vara(id.file, id.b2_tot, start, count, &magEnergy_h[0].x )) ERR(retval);
    if (retval = nc_put_vara(id.file, id.v2_tot, start, count, &kinEnergy_h[0].x )) ERR(retval);
    if (retval = nc_sync(id.file)) ERR(retval);

    free(totEnergy_h); free(kinEnergy_h); free(magEnergy_h); 

    DEBUGPRINT("Exiting energy\n");

}    

void peak(cuComplex* kPhi, float time)
{
    float *rPhi;
    cudaMalloc((void**) &rPhi, sizeof(float)*Nx*Ny*Nz);

    if(cufftExecC2R(plan_C2R, kPhi, rPhi) != CUFFT_SUCCESS) printf("oops in peak \n");

    float *rPhi_h;
    rPhi_h = (float*) malloc(sizeof(float)*Nx*Ny*Nz);
    CP_TO_CPU(rPhi_h, rPhi, sizeof(float)*Nx*Ny*Nz);
    cudaFree(rPhi);

    int idx;
    int i, j, k, isave, jsave, ksave;
    int idxmax;
    float pmax, zpos;
    pmax = 0.;
    isave = 0;
    jsave = 0;
    ksave = 0;

    // find the peak 
    for (k=0; k<Nz; k++) {
        for (i=0; i<1; i++) {
            for (j=0; j<1; j++) {
                idx = j + Ny*i + Ny*Nx*k;
                if (rPhi_h[idx]>pmax)
                {
                    pmax = rPhi_h[idx];
                    isave = i;
                    jsave = j;
                    ksave = k;
                }
            }
        }
    }

    if (nsteps == 0) {
        fprintf (awp_collfile, "# z        Phi \n");
        for (k=0; k<Nz; k++) {
            zpos = k*2.*3.1415*Z0/Nz;
            for (i=0; i<1; i++) {
                for (j=0; j<1; j++) {
                    idx = j + Ny*i + Ny*Nx*k;
                    fprintf (awp_collfile, "%f \t %f \n", zpos, rPhi_h[idx]);
                }
            }
        }
    } else {
        zpos = ksave*2.*3.1415*Z0/Nz;
        fprintf (awp_collfile, "t= %f\t z= %f\t Peak= %f \n", time, zpos, pmax);
    }

    free (rPhi_h);  
}

// Assumes we are passed Psi (not k_perp Psi or somesuch )
void j_z_diag( cuComplex * A, float time, int jstep, struct NetCDF_ids id )
{
    multKPerp <<<dG,dB>>> (A,A,-1.0);

    // Now have -k_perp^2 Psi = (const) * j_z

    if(cufftExecC2R(plan_C2R, A, fdxR ) != CUFFT_SUCCESS) printf("Inverse FFT for diagnostics failed. \n");	

	 size_t Nf = Nx * Ny * Nz;
    float *j_z_data = (float*)malloc( sizeof(float) * Nf );
    CP_TO_CPU( j_z_data, fdxR, sizeof(float) * Nf );

    size_t start[4],count[4];
    start[0] = jstep; start[1] = 0;  start[2] = 0;  start[3] = 0;
    count[1] = 1;     count[1] = Nz; count[2] = Nx; count[3] = Ny;

    int retval;
    if (retval = nc_put_vara(id.file, id.jz, start, count, j_z_data)) ERR(retval);

    free (j_z_data);

    return;
}


//////////////////////////////////////////////////////////////////////
// Main diagnostic calling functions
//////////////////////////////////////////////////////////////////////



void alf_diagnostics(cuComplex* kPhi, cuComplex* kA, cuComplex* zp, cuComplex* zm, float time, int jstep, struct NetCDF_ids id){
    // Calculate kperp**2 * phi and kperp**2 A for all alfven diagnostics
    addsubt <<<dG,dB>>> (kPhi, zp, zm, 1);
    //kPhi = zp+zm

    scale <<<dG,dB>>> (kPhi, .5);
    //kPhi = .5*(zp+zm) = phi

    addsubt <<<dG,dB>>> (kA, zp, zm, -1);
    //kA = zp-zm

    scale <<<dG,dB>>> (kA, .5);
    //kA = .5*(zp-zm) = A

	 // Pass Psi(kx,ky,kz) to the diagnostic
	 j_z_diag( kA, time, jstep, id );
   
    // reset kA = Psi
    addsubt <<<dG,dB>>> (kA, zp, zm, -1);
    scale <<<dG,dB>>> (kA, .5);
    
    if (linonly) peak(kPhi, time);

    squareComplex <<<dG,dB>>> (kPhi);
    //kPhi = phi**2

    squareComplex <<<dG,dB>>> (kA);
    //kA = A**2

    fixFFT <<<dG,dB>>>(kPhi);
    fixFFT <<<dG,dB>>>(kA);

    multKPerp <<<dG,dB>>> (kPhi, kPhi, -1);
    //kPhi = (kperp**2) * (phi**2)

    multKPerp <<<dG,dB>>> (kA, kA, -1);
    //kA = (kperp**2) * (A**2)

    int retval;
    size_t start[1], count[1];
    start[0] = jstep; 
    count[0] = 1;
    if (retval = nc_put_vara(id.file, id.t, start, count, &time)) ERR(retval);

    energy_kz_kperp(kPhi, kA, time, jstep, id);
    energy(kPhi, kA, time, jstep, id);

    /*

    // Alfven wave collision diagnostics
    CP_ON_GPU(kPhi, zp, Nkc);
    CP_ON_GPU(kA, zm, Nkc);

    squareComplex <<<dG,dB>>> (kPhi);
    //kPhi = z+^2

    squareComplex <<<dG,dB>>> (kA);
    //kA = z-^2

    // No fixfft needed. If you called fixfft here, then the ky=0 mode needs to be
    // dealt with specially in the aw_coll_diag subroutine 
    aw_coll_diag(kPhi, kA, time);
     */

}

//////////////////////////////////////////////////////////////////////
// Initialize NetCDF diagnostics
//////////////////////////////////////////////////////////////////////

struct NetCDF_ids init_netcdf_diag(struct NetCDF_ids id){

    char str[255];

    int retval;

    //  int ikpmax = (int) ceil( sqrt( pow((float)(Nx-1)/3, 2) + pow((float)(Ny-1)/3, 2) ) );
    //  float kpmax = ceil( sqrt( pow((float)(Nx-1)/3, 2) + pow((float)(Ny-1)/3, 2) ) );
    int ikpmax = (int) (Nx-1)/3;
    float kpmax = (float) (Nx-1)/3;
    float kpar[Nz], kperp[ikpmax];

    strcpy(str, runname);
    strcat(str, ".nc");

    if (retval = nc_create(str, NC_CLOBBER, &id.file)) ERR(retval);

    if (retval = nc_def_dim(id.file, "nkperp", ikpmax-1,      &id.kperp_dim)) ERR(retval);
    if (retval = nc_def_dim(id.file, "nkz",    Nz,            &id.kz_dim))    ERR(retval);
    if (retval = nc_def_dim(id.file, "nkpar",  Nz,            &id.kpar_dim))  ERR(retval);
    if (retval = nc_def_dim(id.file, "time",   NC_UNLIMITED,  &id.t_dim))     ERR(retval);

    if (retval = nc_def_dim(id.file, "kx",   Nx,      &id.kx_dim))     ERR(retval);
    if (retval = nc_def_dim(id.file, "ky",   Ny/2+1,  &id.ky_dim))     ERR(retval);
    if (retval = nc_def_var(id.file, "kx", NC_FLOAT, 1, &id.kx_dim, &id.kx_vals)) ERR(retval);
    if (retval = nc_def_var(id.file, "ky", NC_FLOAT, 1, &id.ky_dim, &id.ky_vals)) ERR(retval);

    if (retval = nc_def_dim(id.file, "x",   Nx,  &id.x_dim))     ERR(retval);
    if (retval = nc_def_dim(id.file, "y",   Ny,  &id.y_dim))     ERR(retval);
    if (retval = nc_def_dim(id.file, "z",   Nz,  &id.z_dim))     ERR(retval);
    if (retval = nc_def_var(id.file, "x", NC_FLOAT, 1, &id.x_dim, &id.x_vals)) ERR(retval);
    if (retval = nc_def_var(id.file, "y", NC_FLOAT, 1, &id.y_dim, &id.y_vals)) ERR(retval);
    if (retval = nc_def_var(id.file, "z", NC_FLOAT, 1, &id.z_dim, &id.z_vals)) ERR(retval);



    static char title[] = "Gandalf simulation data";
    if (retval = nc_put_att_text(id.file, NC_GLOBAL, "Title", strlen(title), title)) ERR(retval);

    time_t time_now;
    char* time_txt;

    time_now = time(NULL);
    time_txt = ctime(&time_now);

    if (retval = nc_put_att_text(id.file, NC_GLOBAL, "Date", strlen(time_txt)-1, time_txt)) ERR(retval);

    if (retval = nc_def_var(id.file, "GPU", NC_INT, 0, 0, &id.gpu)) ERR(retval);
    if (retval = nc_def_var(id.file, "restart", NC_INT, 0, 0, &id.restart)) ERR(retval);

    if (retval = nc_def_var(id.file, "Nx", NC_INT, 0, 0, &id.nx)) ERR(retval);
    if (retval = nc_def_var(id.file, "Ny", NC_INT, 0, 0, &id.ny)) ERR(retval);
    if (retval = nc_def_var(id.file, "Nz", NC_INT, 0, 0, &id.nz)) ERR(retval);

    if (retval = nc_def_var(id.file, "x0", NC_FLOAT, 0, 0, &id.x0)) ERR(retval);
    if (retval = nc_def_var(id.file, "y0", NC_FLOAT, 0, 0, &id.y0)) ERR(retval);
    if (retval = nc_def_var(id.file, "z0", NC_FLOAT, 0, 0, &id.z0)) ERR(retval);

    if (retval = nc_def_var(id.file, "nsteps", NC_INT, 0, 0, &id.nsteps)) ERR(retval);
    if (retval = nc_def_var(id.file, "t", NC_FLOAT, 1, &id.t_dim, &id.t)) ERR(retval);
    // should put text attribute "Time"
    // should put text attribute "Units"

    if (retval = nc_def_var(id.file, "nwrite", NC_INT, 0, 0, &id.nwrite)) ERR(retval);
    if (retval = nc_def_var(id.file, "nforce", NC_INT, 0, 0, &id.nforce)) ERR(retval);
    if (retval = nc_def_var(id.file, "maxdt", NC_FLOAT, 0, 0, &id.maxdt)) ERR(retval);
    if (retval = nc_def_var(id.file, "cfl", NC_FLOAT, 0, 0, &id.cfl)) ERR(retval);
    //  if (retval = nc_def_var(id.file, "restart_name", NC_CHAR, ?, ?, &restart_name)) ERR(retval);

    if (retval = nc_def_var(id.file, "alpha_z", NC_INT, 0, 0, &id.alpha_z)) ERR(retval);
    if (retval = nc_def_var(id.file, "alpha_hyper", NC_INT, 0, 0, &id.alpha_hyper)) ERR(retval);

    if (retval = nc_def_var(id.file, "nu_kz", NC_FLOAT, 0, 0, &id.nu_kz)) ERR(retval);
    if (retval = nc_def_var(id.file, "nu_hyper", NC_FLOAT, 0, 0, &id.nu_hyper)) ERR(retval);

    if (retval = nc_def_var(id.file, "kperp", NC_FLOAT, 1, &id.kperp_dim, &id.kperp)) ERR(retval);
    //  if (retval = nc_def_var(id.file, "kz",    NC_FLOAT, 1, &id.kz_dim,    &id.kz))    ERR(retval);
    if (retval = nc_def_var(id.file, "kpar",  NC_FLOAT, 1, &id.kpar_dim,  &id.kpar))  ERR(retval);

    if (retval = nc_def_var(id.file, "kpeak", NC_INT, 0, 0, &id.kpeak)) ERR(retval);

    if (driven) {
        if (retval = nc_def_dim(id.file, "nkstir",  nkstir, &id.stir_dim)) ERR(retval);
        if (retval = nc_def_var(id.file, "fampl", NC_FLOAT, 0, 0, &id.fampl)) ERR(retval);

        id.kstir[0] = id.stir_dim;

        if (retval = nc_def_var(id.file, "kstir_x", NC_INT, 1, id.kstir, &id.kstir_x)) ERR(retval);
        if (retval = nc_def_var(id.file, "kstir_y", NC_INT, 1, id.kstir, &id.kstir_y)) ERR(retval);
        if (retval = nc_def_var(id.file, "kstir_z", NC_INT, 1, id.kstir, &id.kstir_z)) ERR(retval);
    }

    id.kperps[0] = id.t_dim;
    id.kperps[1] = id.kperp_dim;

    id.kpars[0] = id.t_dim;
    id.kpars[1] = id.kpar_dim;

    id.kparperp[0] = id.t_dim;
    id.kparperp[1] = id.kperp_dim;
    id.kparperp[2] = id.kpar_dim;

    if (retval = nc_def_var(id.file, "b2_kparkperp", NC_FLOAT, 3, id.kparperp, &id.b2)) ERR(retval);
    if (retval = nc_def_var(id.file, "v2_kparkperp", NC_FLOAT, 3, id.kparperp, &id.v2)) ERR(retval);

    if (retval = nc_def_var(id.file, "v2", NC_FLOAT, 1, &id.t_dim, &id.v2_tot )) ERR(retval);
    if (retval = nc_def_var(id.file, "b2", NC_FLOAT, 1, &id.t_dim, &id.b2_tot )) ERR(retval);

    id.txyz[0] = id.t_dim;
    id.txyz[1] = id.z_dim;
    id.txyz[2] = id.x_dim;
    id.txyz[3] = id.y_dim;

    if (retval = nc_def_var(id.file, "jz", NC_FLOAT, 4, id.txyz, &id.jz)) ERR(retval);

    if (retval = nc_enddef(id.file)) ERR(retval);

    // need to define a temporary array for kpar (which will also serve for kz because kz is defined only as a function)

    for (int ikz=0; ikz<Nz; ikz++) {kpar[ikz] = kz(ikz);}
    for (int ikp=0; ikp<ikpmax; ikp++) {kperp[ikp] = ((float) ikp/ikpmax)*kpmax;} // only makes sense if X0 and Y0 each are unity. BD
    if (retval = nc_put_var(id.file, id.kpar,  kpar))  ERR(retval); 
    if (retval = nc_put_var(id.file, id.kperp, kperp)) ERR(retval); 

	 float kx_vals[Nx],ky_vals[ Ny/2 + 1 ];
	 for (int ikx=0; ikx < Nx; ++ikx) { kx_vals[ ikx ] = kx( ikx ); };
	 for (int iky=0; iky < Ny/2 + 1; ++iky) { ky_vals[ iky ] = ky( iky ); };
    if (retval = nc_put_var(id.file, id.kx_vals, kx_vals)) ERR(retval); 
    if (retval = nc_put_var(id.file, id.ky_vals, ky_vals)) ERR(retval); 

    float x_vals[Nx],y_vals[Ny],z_vals[Nz];
	 for (int ix=0; ix < Nx; ++ix) { x_vals[ ix ] = static_cast<float>( ix )/( Nx * X0 ); };
    if (retval = nc_put_var(id.file, id.x_vals, x_vals)) ERR(retval); 
	 for (int iy=0; iy < Ny; ++iy) { y_vals[ iy ] = static_cast<float>( iy )/( Ny * Y0 ); };
    if (retval = nc_put_var(id.file, id.y_vals, y_vals)) ERR(retval); 
	 for (int iz=0; iz < Nz; ++iz) { z_vals[ iz ] = static_cast<float>( iz )/( Nz * Z0 ); };
    if (retval = nc_put_var(id.file, id.z_vals, z_vals)) ERR(retval); 


    if (retval = nc_put_var(id.file, id.nx, &Nx)) ERR(retval);
    if (retval = nc_put_var(id.file, id.ny, &Ny)) ERR(retval);
    if (retval = nc_put_var(id.file, id.nz, &Nz)) ERR(retval);

    if (retval = nc_put_var(id.file, id.x0, &X0)) ERR(retval);
    if (retval = nc_put_var(id.file, id.y0, &Y0)) ERR(retval);
    if (retval = nc_put_var(id.file, id.z0, &Z0)) ERR(retval);

    if (retval = nc_put_var(id.file, id.nsteps, &nsteps)) ERR(retval);
    if (retval = nc_put_var(id.file, id.nwrite, &nwrite)) ERR(retval);
    if (retval = nc_put_var(id.file, id.nforce, &nforce)) ERR(retval);
    if (retval = nc_put_var(id.file, id.maxdt, &maxdt)) ERR(retval);
    if (retval = nc_put_var(id.file, id.cfl, &cfl)) ERR(retval);

    if (retval = nc_put_var(id.file, id.alpha_z, &alpha_z)) ERR(retval);
    if (retval = nc_put_var(id.file, id.alpha_hyper, &alpha_hyper)) ERR(retval);

    if (retval = nc_put_var(id.file, id.nu_kz, &nu_kz)) ERR(retval);
    if (retval = nc_put_var(id.file, id.nu_hyper, &nu_hyper)) ERR(retval);

    if (retval = nc_put_var(id.file, id.kpeak, &kpeak)) ERR(retval);

    if (driven) {
        if (retval = nc_put_var(id.file, id.nkstir, &nkstir)) ERR(retval);
        if (retval = nc_put_var(id.file, id.fampl, &fampl)) ERR(retval);

        size_t s[1];
        size_t c[1];

        s[0] = 0;
        c[0] = nkstir;

        if (retval = nc_put_vara(id.file, id.kstir_x, s, c, &kstir_x[0])) ERR(retval);
        if (retval = nc_put_vara(id.file, id.kstir_y, s, c, &kstir_y[0])) ERR(retval);
        if (retval = nc_put_vara(id.file, id.kstir_z, s, c, &kstir_z[0])) ERR(retval);
    }

    return id;
}


//////////////////////////////////////////////////////////////////////
// Initialize diagnostics
//////////////////////////////////////////////////////////////////////
void init_diag(){

    char str[255];

    /////////////////////////////////////////////////////////////////
    //Alfven diagnostics
    /////////////////////////////////////////////////////////////////
    //  strcpy(str, runname);
    //  strcat(str, ".energy");
    //  energyfile = fopen( str, "w+");
    //  strcpy(str, runname);
    //  strcat(str, ".speckzkp");
    //  alf_kzkpfile = fopen( str, "w+");

    /*  
    /////////////////////////////////////////////////////////////////
    //Alfven wave collision diagnostic
    /////////////////////////////////////////////////////////////////
    strcpy(str, runname);
    strcat(str, ".awcollp");
    awp_collfile = fopen( str, "w+");

    strcpy(str, runname);
    strcat(str, ".awcollm");
    awm_collfile = fopen( str, "w+");
     */

    strcpy(str, runname);
    strcat(str, ".awcollp");
    awp_collfile = fopen( str, "w+");
}
//////////////////////////////////////////////////////////////////////
void close_diag(){

    //  fclose(energyfile);
    //  fclose(alf_kzkpfile);

    fclose(awp_collfile);
    //  fclose(awm_collfile);


}

void close_netcdf_diag(struct NetCDF_ids id){

    int retval;

    if (retval= nc_close(id.file)) ERR(retval);

}

void diagnostics(cuComplex* zp, cuComplex* zm, float time, int jstep, struct NetCDF_ids id){

    alf_diagnostics(temp1, temp2, zp, zm, time, jstep, id);
}
//////////////////////////////////////////////////////////////////////
// Alfven collision diagnostic
//////////////////////////////////////////////////////////////////////
void aw_coll_diag(cuComplex* zp2, cuComplex* zm2, float time){

    int ikx, iky, ikz, index;

    cuComplex *energy;
    energy = (cuComplex*) malloc(sizeof(cuComplex)*2); 

    // Primary mode (1, 0, -1) copied over to energy[0]
    ikx = 1; iky=0; ikz = Nz-1;
    index = iky+(Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;
    CP_TO_CPU(energy, zp2 + index, sizeof(cuComplex));

    // Primary mode (0, 1, 1) copied over to energy[1]
    ikx = 0; iky=1; ikz = 1;
    index = iky+(Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;
    CP_TO_CPU(energy+1, zp2 + index, sizeof(cuComplex));

    // Primary modes written out. The ky=0 mode gets scaled up due to reality condition
    fprintf(awp_collfile, "%g \t %g \t", time, sqrt(energy[0].x+energy[1].x));

    // Secondary mode (1, 1, 0) copied over to energy[0]
    ikx = 1; iky=1; ikz = 0;
    index = iky+(Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;
    CP_TO_CPU(energy, zp2 + index, sizeof(cuComplex));

    //Secondary mode written out
    fprintf(awp_collfile, "%g \t ", sqrt(energy[0].x));

    // Tertiary mode (2, 1, -1) copied over to energy[0]
    ikx = 2; iky=1; ikz = Nz-1;
    index = iky+(Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;
    CP_TO_CPU(energy, zp2 + index, sizeof(cuComplex));

    // Tertiary mode (1, 2, 1) copied over to energy[1]
    ikx = 1; iky=2; ikz = 1;
    index = iky+(Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;
    CP_TO_CPU(energy+1, zp2 + index, sizeof(cuComplex));

    // Tertiary mode written out
    fprintf(awp_collfile, "%g\n", sqrt(energy[0].x + energy[1].x));

    fflush(awp_collfile);

    // Primary mode (1, 0, -1) copied over to energy[0]
    ikx = 1; iky=0; ikz = Nz-1;
    index = iky+(Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;
    CP_TO_CPU(energy, zm2 + index, sizeof(cuComplex));

    // Primary mode (0, 1, 1) copied over to energy[1]
    ikx = 0; iky=1; ikz = 1;
    index = iky+(Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;
    CP_TO_CPU(energy+1, zm2 + index, sizeof(cuComplex));

    // Primary modes written out. The ky=0 mode gets scaled up due to reality condition
    fprintf(awm_collfile, "%g \t %g \t", time, sqrt(energy[0].x+energy[1].x));

    // Secondary mode (1, 1, 0) copied over to energy[0]
    ikx = 1; iky=1; ikz = 0;
    index = iky+(Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;
    CP_TO_CPU(energy, zm2 + index, sizeof(cuComplex));

    //Secondary mode written out
    fprintf(awm_collfile, "%g \t ", sqrt(energy[0].x));

    // Tertiary mode (2, 1, -1) copied over to energy[0]
    ikx = 2; iky=1; ikz = Nz-1;
    index = iky+(Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;
    CP_TO_CPU(energy, zm2 + index, sizeof(cuComplex));

    // Tertiary mode (1, 2, 1) copied over to energy[1]
    ikx = 1; iky=2; ikz = 1;
    index = iky+(Ny/2+1)*ikx + (Ny/2+1)*Nx*ikz;
    CP_TO_CPU(energy+1, zm2 + index, sizeof(cuComplex));

    // Tertiary mode written out
    fprintf(awm_collfile, "%g\n", sqrt(energy[0].x + energy[1].x));

    fflush(awm_collfile);

    free(energy);


}
