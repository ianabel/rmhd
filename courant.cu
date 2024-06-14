void courant(float* dt,  cuComplex* zp, cuComplex* zm)
{
  zero <<<dG, dB>>> (padded, Nx, Ny, Nz);

  cuComplex *max;
  max = (cuComplex*) malloc(sizeof(cuComplex));
    
  float vxmax, vymax, omega_zmax;
    
  vxmax = 0.f;
  vymax = 0.f;

  // calculate max velocity_x_plus = max(ky*zp)
    
  // temp1 holds velocity = ky*zp_plus

  multKy <<<dG,dB>>> (temp1, zp);
  CUDA_DEBUG("multKy zp: %s\n");

  maxReduc(max, temp1, padded); 

  vxmax = sqrt(max[0].x*max[0].x+max[0].y*max[0].y);		        
    
  // calculate max velocity_x_minus (ky*zm) = max vx_minus
    
  // temp1 = ky*zm
  multKy <<<dG,dB>>> (temp1,zm);
  CUDA_DEBUG("multKy zm: %s\n");
    
  maxReduc(max,temp1,padded);
    
  if(sqrt(max[0].x*max[0].x+max[0].y*max[0].y) > vxmax) {
    vxmax = sqrt(max[0].x*max[0].x+max[0].y*max[0].y);
  }  
   
  //////////////////////////////////////////////////////////
    
  //calculate max(kx*zp)
    
  multKx <<<dG,dB>>> (temp1, zp);
  CUDA_DEBUG("multKx zp: %s\n");
  
  //temp1 = kx*zp
    
  maxReduc(max,temp1,padded);
    
  vymax = sqrt(max[0].x*max[0].x+max[0].y*max[0].y);
    		     
  ///////////////////////////////////////////////////////
    
  //calculate max(kx*zm)
    
  multKx <<<dG,dB>>> (temp1,zm);
  CUDA_DEBUG("multKx zm: %s\n");
  
  //temp1 = kx*zm
    
  maxReduc(max,temp1,padded);
    
  if( sqrt(max[0].x*max[0].x+max[0].y*max[0].y) > vymax) {
    vymax = sqrt(max[0].x*max[0].x+max[0].y*max[0].y);
  }  
    
  /////////////////////////////////////////////////////////
  // omega_zmax
  omega_zmax =  ((float) (Nz-1)/3)/(Z0*cfl);

  /////////////////////////////////////////////////////////
    
  //find dt

  if (vxmax==vxmax || vymax==vymax) {
    if(vxmax>=vymax) *dt = (float) cfl *M_PI*X0/(vxmax*Nx);
    else *dt = (float) cfl*M_PI*Y0/(vymax*Ny);
  }
  //  if(1.0/(*dt) <  omega_zmax) *dt = 1.0f/(omega_zmax);
  if(*dt >  maxdt) *dt = maxdt;

  /*
    printf("dt = %f\n", *dt);
    printf("vxmax = %f\n", vxmax);
    printf("vymax = %f\n", vymax);
    printf("\n");
    if (*dt < 1.e-5) {
    printf("dt = %e, vxmax = %e, vymax = %e\n", *dt, vxmax, vymax);
    } 
  */
  assert(dt=dt);

  free(max);
  DEBUGPRINT("Exiting courant.cu dt = %f\n", *dt);

  // temp1 is free now
    
}

