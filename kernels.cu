/*
__device__ unsigned int get_idx(void) {return __umul24(blockIdx.x,blockDim.x)+threadIdx.x;}
__device__ unsigned int get_idy(void) {return __umul24(blockIdx.y,blockDim.y)+threadIdx.y;}
__device__ unsigned int get_idz(void) {return __umul24(blockIdx.z,blockDim.z)+threadIdx.z;}

__device__ int iget_idx(void) {return __umul24(blockIdx.x,blockDim.x)+threadIdx.x;}
__device__ int iget_idy(void) {return __umul24(blockIdx.y,blockDim.y)+threadIdx.y;}
__device__ int iget_idz(void) {return __umul24(blockIdx.z,blockDim.z)+threadIdx.z;}
*/

__device__ unsigned int get_idx(void) {return blockIdx.x*blockDim.x+threadIdx.x;}
__device__ unsigned int get_idy(void) {return blockIdx.y*blockDim.y+threadIdx.y;}
__device__ unsigned int get_idz(void) {return blockIdx.z*blockDim.z+threadIdx.z;}

__device__ int iget_idx(void) {return blockIdx.x*blockDim.x+threadIdx.x;}
__device__ int iget_idy(void) {return blockIdx.y*blockDim.y+threadIdx.y;}
__device__ int iget_idz(void) {return blockIdx.z*blockDim.z+threadIdx.z;}


__host__ __device__ cuComplex operator+(cuComplex f, cuComplex g) 
{
  return cuCaddf(f,g);
} 

__host__ __device__ cuComplex operator-(cuComplex f, cuComplex g)
{
  return cuCsubf(f,g);
}  

__host__ __device__ cuComplex operator*(float scaler, cuComplex f) 
{
  cuComplex result;
  result.x = scaler*f.x;
  result.y = scaler*f.y;
  return result;
}

__host__ __device__ cuComplex operator*(cuComplex f, float scaler) 
{
  cuComplex result;
  result.x = scaler*f.x;
  result.y = scaler*f.y;
  return result;
}

__host__ __device__ cuComplex operator*(cuComplex f, cuComplex g)
{
  return cuCmulf(f,g);
}

__host__ __device__ cuComplex operator/(cuComplex f, float scaler)
{
  cuComplex result;
  result.x = f.x / scaler;
  result.y = f.y / scaler;
  return result;
}

__host__ __device__ cuComplex operator/(cuComplex f, cuComplex g) 
{
  return cuCdivf(f,g);
}


__host__ __device__ cuComplex exp(cuComplex arg)
{
  cuComplex res;
  float s, c;
  float e = expf(arg.x);
  sincosf(arg.y, &s, &c);
  res.x = c * e;
  res.y = s * e;
  return res;
}

__host__ __device__ cuComplex pow(cuComplex arg, int power)
{
  cuComplex res;
  float r = sqrt(pow(arg.x,2) + pow(arg.y,2));
  float theta = M_PI/2.0;
  if(arg.x != 0.0) theta = atan(arg.y/arg.x);
  res.x = pow(r, power) * cos(power * theta);
  res.y = pow(r, power) * sin(power * theta);
  return res;
}

__host__ __device__ cuComplex conjg(cuComplex arg)
{
  cuComplex conjugate;
  conjugate.x = arg.x;
  conjugate.y = -arg.y;
  return conjugate;

}

__host__ __device__ int sgn(float k) {
	
	if(k>0) return 1;
	if(k<0) return -1;
	return 0;

}

// Wavenumber functions
__host__ __device__ float kx(int ikx)
{
	if(ikx<Nx/2 +1) return ikx/X0;
	else return (ikx-Nx)/X0;

}
__host__ __device__ float ky(int iky)
{
	if(iky<Ny/2+1) return (float) iky/Y0;
	else return 0;
}
__host__ __device__ float kz(int ikz)
{
	if(ikz<Nz/2 +1) return ikz/Z0;
	else return (ikz-Nz)/Z0;

}
// Real space functions
__host__ __device__ float xx(int ix)
{
	if(ix<Nx) return ((float) ix/Nx) *2.0f*M_PI*X0;
	else return -1.0f;
}

__host__ __device__ float yy(int iy)
{
	if(iy<Ny) return ((float) iy/Ny) *2.0f*M_PI*Y0;
	else return -1.0f;
}

__host__ __device__ float zz(int iz)
{
	if(iz<Nz) return ((float) iz/Nz) *2.0f*M_PI*Z0;
	else return -1.0f;
}

// kPerp2 functions
__host__ __device__ float kPerp2(int ikx, int iky)
{
		float kp2 = -kx(ikx)*kx(ikx) -ky(iky)*ky(iky);
		return kp2;
}

__host__ __device__ float kPerp2Inv(int ikx, int iky)
{
		float kp2 = -kx(ikx)*kx(ikx) -ky(iky)*ky(iky);
		if(ikx !=0 || iky !=0) return 1.0f/kp2;
		else return 0.0f;
}

//////////////////////
// Damp in z 
//////////////////////
__global__ void dampz(cuComplex* znew, float nu_kz, int alpha_z, float dt)
{
  if(Nz>1){
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
	  int idmax = (Nz-1)/3;

		znew[index] = znew[index] *exp(- nu_kz*dt*pow(abs(kz(idz)/kz(idmax)),2*alpha_z));
		}
		}
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;

		int IDZ = idz + zThreads*i;
		int idmax = (Nz-1)/3;

		znew[index] = znew[index] *exp(- nu_kz*dt*pow(abs(kz(IDZ)/kz(idmax)),2*alpha_z));
	      }
	    }
	  }
	}

}
	
//////////////////////
// Kperp Damp 
//////////////////////
__global__ void damp_hyper(cuComplex* znew, float nu_hyper, int alpha_hyper, float dt)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
	  int idxmax = (Nx-1)/3;
	  int idymax = (Ny-1)/3;

		znew[index] = znew[index] *exp(-nu_hyper*dt*pow(abs(kPerp2(idx,idy)/kPerp2(idxmax,idymax)),alpha_hyper));
		}
	}
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	    int idxmax = (Nx-1)/3;
	    int idymax = (Ny-1)/3;

		znew[index] = znew[index] *exp(-nu_hyper*dt*pow(abs(kPerp2(idx,idy)/kPerp2(idxmax,idymax)),alpha_hyper) );
     }
    }
   }
}


// kp shell
__global__ void kpshellsum(cuComplex* k2field2, int ikp, float* temp)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  if(Nz<=zThreads) {
    if(idx<Nx && idy<Ny/2+1 && idz<Nz){ 
      if(sqrt(abs(kPerp2(idx,idy))) >= ikp-1.0 && 
	 sqrt(abs(kPerp2(idx,idy))) < ikp+1.0)  {

	int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
	atomicAdd(temp + ikp, k2field2[index].x);
      }
    }
  }
  else{
    for(int i=0; i<Nz/zThreads; i++){
      if(idx<Nx && idy<Ny/2+1 && idz<zThreads){ 
	int IDZ = idz + zThreads*i;
	if(sqrt(abs(kPerp2(idx,idy))) >= ikp-1.0 && 
	   sqrt(abs(kPerp2(idx,idy))) < ikp+1.0)  {

	  int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*IDZ;
	  atomicAdd(temp + ikp, k2field2[index].x);

	}
      }
    }
  }
}

__global__ void kpshellsum(float* k2field2, int ikp, float* temp)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  if(Nz<=zThreads) {
    if(idx<Nx && idy<Ny/2+1 && idz<Nz){ 
      if(sqrt(abs(kPerp2(idx,idy))) >= ikp-1.0 && 
	 sqrt(abs(kPerp2(idx,idy))) < ikp+1.0)  {

	int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
	atomicAdd(temp + ikp, k2field2[index]);
      }
    }
  }
  else{
    for(int i=0; i<Nz/zThreads; i++){
      if(idx<Nx && idy<Ny/2+1 && idz<zThreads){ 
	int IDZ = idz + zThreads*i;
	if(sqrt(abs(kPerp2(idx,idy))) >= ikp-1.0 && 
	   sqrt(abs(kPerp2(idx,idy))) < ikp+1.0)  {

	  int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*IDZ;
	  atomicAdd(temp + ikp, k2field2[index]);

	}
      }
    }
  }

}
// kz-kp shell
__global__ void kz_kpshellsum(cuComplex* k2field2, int ikp, float* energy_kp)
{

  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  // X0 and Y0 needs to be 1 for this to make sense.
  int ikz_kp;

  if(Nz<=zThreads) {
    if(idx<Nx && idy<Ny/2+1 && idz<Nz){
      if(sqrt(abs(kPerp2(idx,idy))) >= ikp-0.5 && sqrt(abs(kPerp2(idx,idy))) < ikp+0.5 )  {
	unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
	ikz_kp = idz + Nz*ikp;
	atomicAdd(energy_kp + ikz_kp, k2field2[index].x);
      }
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idx<Nx && idy<Ny/2+1 && idz<zThreads) {
	if(sqrt(abs(kPerp2(idx,idy))) >= ikp-0.5 && sqrt(abs(kPerp2(idx,idy))) < ikp+0.5)  {
	  unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	  int IDZ = idz + zThreads*i;
	  ikz_kp = IDZ + Nz*ikp;
	  atomicAdd(energy_kp + ikz_kp, k2field2[index].x);
	}
      }
    }
  }
}
__global__ void kz_kpshellsum(float* k2field2, int ikp, float* energy_kp)
{

  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  // X0 and Y0 needs to be 1 for this to make sense.
  int ikz_kp;

  if(Nz<=zThreads) {
    if(idx<Nx/2+1 && idz<Nz){
      if(sqrt(abs(kPerp2(idx,idy))) >= ikp-0.5 && sqrt(abs(kPerp2(idx,idy))) < ikp+0.5)  {
	unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
	ikz_kp = idz + Nz*ikp;
	atomicAdd(energy_kp + ikz_kp, k2field2[index]);
      }
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idx<Nx/2+1 && idz<zThreads) {
	if(sqrt(abs(kPerp2(idx,idy))) >= ikp-0.5 && sqrt(abs(kPerp2(idx,idy))) < ikp+0.5)  {
	  unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	  int IDZ = idz + zThreads*i;
	  ikz_kp = IDZ + Nz*ikp;
	  atomicAdd(energy_kp + ikz_kp, k2field2[index]);
	}
      }
    }
  }
}

__global__ void fft_interp(float* result, cuComplex* function, float* xx, float* yy, float zz)
{

  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  cuComplex cj; cj.x = 0.0f; cj.y = 1.0f;
  cuComplex tmp;
  if(Nz<=zThreads){
    if(idx<Nx && idy<Ny/2+1 && idz<Nz){
      int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      tmp = function[index] * exp(cj*kx(idx)*xx[0] + cj*ky(idy)*yy[0] + cj*kz(idz)*zz);
      if(idy!=0) tmp = tmp*2.0f;
      atomicAdd(result, tmp.x);
    }
  }
  else{
    for(int i=0; i<Nz/zThreads; i++){
      if(idx<Nx && idy<Ny/2+1 && idz<zThreads){
	int IDZ = idz + zThreads*i;
	int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*IDZ;
	tmp = function[index] * exp(cj*kx(idx)*xx[0] + cj*ky(idy)*yy[0] + cj*kz(IDZ)*zz);
	if(idy!=0) tmp = tmp*2.0f;
	atomicAdd(result, tmp.x);
      }
    }
  }
}


__global__ void fldtracestep(float* cnew, float* cold,  float* dBfield, float dz)
{
  cnew[0] = cold[0] + dBfield[0]*dz;
  cnew[0] = cnew[0] - 2.0*M_PI*((float) floor(cnew[0]/(2.0*M_PI) + 0.0*0.5));
}

__global__ void assign_fld(float* xfld, float* yfld, float *xfld0, float *yfld0)
{
  xfld[0] = xfld0[0];
  yfld[0] = yfld0[0];
}

__global__ void assign_foot(float* xfld0, float* yfld0)
{
  unsigned int idx = get_idx();
  if(idx<Nz){
    xfld0[idx] = 0.0;
    yfld0[idx] = 1.0;
  }
}

__global__ void sl_k_damp_calc(float *Gmdamp, cuComplex* Gm, float nu_kp_g, int alpha_kp_g, float nu_kz_g, int alpha_kz_g, int m){
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      int idxmax = (Nx-1)/3; int idymax = (Ny-1)/3; int idzmax = (Nz-1)/3;

      Gmdamp[index] = (Gm[index].x*Gm[index].x + Gm[index].y*Gm[index].y)
	*(1.0f - exp(-2.0f*nu_kp_g*pow(abs(kPerp2(idx,idy)/kPerp2(idxmax,idymax)),alpha_kp_g))
	  *exp(-2.0f*nu_kz_g*pow(abs(kz(idz)/kz(idzmax)),2*alpha_kz_g)));
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	int idxmax = (Nx-1)/3; int idymax = (Ny-1)/3; int idzmax = (Nz-1)/3;
	int IDZ = idz + zThreads*i;

	Gmdamp[index] = (Gm[index].x*Gm[index].x + Gm[index].y*Gm[index].y)
	  *(1.0f - exp(-2.0f*nu_kp_g*pow(abs(kPerp2(idx,idy)/kPerp2(idxmax,idymax)),alpha_kp_g))
	    *exp(-2.0f*nu_kz_g*pow(abs(kz(IDZ)/kz(idzmax)),2*alpha_kz_g)));
      }
    }
  }

}

__global__ void kInit(float* kx, float* ky, float* kz) 
{
  int idx = iget_idx();
  int idz = iget_idz();
  unsigned int idy = get_idy();

  if(idy<Ny/2+1 && idx<Nx) {
      
    ky[idy] = (float) idy/Y0;
      
    if(idx<Nx/2+1) {					
      kx[idx] = (float) idx/X0;					
    } else {						
      kx[idx] = (float) (idx - Nx)/X0;				
    }
  }
  
  if(Nz<=zThreads) { 
    if(idz<Nz) {
      if(idz<(Nz/2+1))
        kz[idz] = (float) idz/Z0;
      else
        kz[idz] = (float) (idz - Nz)/Z0;
    }	
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idz<zThreads) {
	int IDZ = idz + zThreads*i;
	if(IDZ<Nz){
	if(IDZ<(Nz/2+1))
	  kz[IDZ] = (float) IDZ/Z0;
	else
	  kz[IDZ] = (float) (IDZ - Nz)/Z0;
      }
	  }
    }
  } 
}     

__global__ void kPerpInit(float* kPerp2, float* kx, float* ky)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
   
  if(idy<(Ny/2+1) && idx<Nx) {
    unsigned int index = idy + (Ny/2+1)*idx;
      
    kPerp2[index] = -kx[idx]*kx[idx] -ky[idy]*ky[idy]; 	     
  }
}   

__global__ void kPerpInvInit(float* kPerp2Inv, float* kPerp2)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  
  if(idy<(Ny/2+1) && idx<Nx) {
    unsigned int index = idy + (Ny/2+1)*idx;
      
    if(index !=0) kPerp2Inv[index] = (float) 1.0f / (kPerp2[index]);  
    
  }
  kPerp2Inv[0] = 0.0;
}   
__global__ void deriv(cuComplex* f, cuComplex* fdx, cuComplex* fdy)                        
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  cuComplex cj; cj.x = 0.0f; cj.y = 1.0f;
  
  if(Nz<=zThreads) {
   if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
     unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
     //df/dy
     fdy[index] = cj * ky(idy) * f[index];
    
     //df/dx
     fdx[index] = cj * kx(idx) * f[index];
   }
  } 
  else {
   for(int i=0; i<Nz/zThreads; i++) { 
    if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
    unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
    
    //df/dx
    fdy[index] = cj * ky(idy) * f[index];
    
    //df/dy
    fdx[index] = cj * kx(idx) * f[index];
    }
   }
  } 
}  

__global__ void mask(cuComplex* mult) 
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  if(Nz<=zThreads) {
   if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
    unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
        
    if( (idy>(Ny-1)/3  || (idx>(Nx-1)/3 && idx<2*Nx/3+1) || (idz>(Nz-1)/3 && idz<2*Nz/3+1) ) ) {
      mult[index].x = 0.0f;
      mult[index].y = 0.0f;
    }  
   }
  }
  else {
   for(int i=0; i<Nz/zThreads; i++) {
    if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
     unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	 int IDZ = idz + zThreads*i;
    
     if( (idy>(Ny-1)/3  || (idx>(Nx-1)/3 && idx<2*Nx/3+1) || (IDZ>(Nz-1)/3 && IDZ<2*Nz/3+1) ) ) {
       mult[index].x = 0.0f;
       mult[index].y = 0.0f;
     }  
    }
   }
  }
}      
  
  

__global__ void bracket(float* mult, float* fdx, float* fdy, 
                      float* gdx, float* gdy, float scaler)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  if(Nz<=zThreads) {
   if(idy<(Ny) && idx<Nx && idz<Nz ) {
    unsigned int index = idy + (Ny)*idx + Nx*(Ny)*idz;

    mult[index] = scaler*( (fdx[index])*(gdy[index]) - (fdy[index])*(gdx[index]) );  
   }
  }
  else {
   for(int i=0; i<Nz/zThreads; i++) {
    if(idy<(Ny) && idx<Nx && idz<zThreads ) {
    unsigned int index = idy + (Ny)*idx + Nx*(Ny)*idz + Nx*Ny*zThreads*i;
    
    mult[index] = scaler*( (fdx[index])*(gdy[index]) - (fdy[index])*(gdx[index]) );  
    }
   }
  } 
 
}  
__global__ void sum(cuComplex* result, cuComplex* a)
{
  //shared mem size = 8*8*8*sizeof(cuComplex)
  extern __shared__ cuComplex result_s[];
  //tid up to blockDim.x*blockDim.y*blockDim.z = 8*8*8
  int tid = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
    
  if(tid<8*8*8) {
    result_s[tid].x = 0;
    result_s[tid].y = 0;
    
    result_s[tid] = a[blockIdx.x*blockDim.x*blockDim.y*blockDim.z+tid];
    __syncthreads();
    
    for(int s=(blockDim.x*blockDim.y*blockDim.z)/2; s>0; s>>=1) {
      if(tid<s) {
        result_s[tid].x += result_s[tid+s].x;	
		result_s[tid].y += result_s[tid+s].y;
      }
      __syncthreads();
    }
    
    if(tid==0) {
      result[blockIdx.x].x = result_s[0].x;
      result[blockIdx.x].y = result_s[0].y;
    }   
  }
}

__global__ void sum(float* result, float* a)
{
  //shared mem size = 8*8*8*sizeof(cuComplex)
  extern __shared__ float result_s_real[];
  //tid up to blockDim.x*blockDim.y*blockDim.z = 8*8*8
  int tid = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
    
  if(tid<8*8*8) {
    result_s_real[tid] = 0;
    
    result_s_real[tid] = a[blockIdx.x*blockDim.x*blockDim.y*blockDim.z+tid];
    __syncthreads();
    
    for(int s=(blockDim.x*blockDim.y*blockDim.z)/2; s>0; s>>=1) {
      if(tid<s) {
        result_s_real[tid] += result_s_real[tid+s];	
      }
      __syncthreads();
    }
    
    if(tid==0) {
      result[blockIdx.x] = result_s_real[0];
    }   
  }
}

__global__ void maximum(cuComplex* result, cuComplex* a)
{
  //shared mem size = 8*8*8*sizeof(cuComplex)
  extern __shared__ cuComplex result_s[];
  //tid up to blockDim.x*blockDim.y*blockDim.z = 8*8*8
  int tid = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
    
  if(tid<8*8*8) {
    result_s[tid].x = 0;
    result_s[tid].y = 0;
    
    result_s[tid] = a[blockIdx.x*blockDim.x*blockDim.y*blockDim.z+tid];
    __syncthreads();
    
    for(int s=(blockDim.x*blockDim.y*blockDim.z)/2; s>0; s>>=1) {
      if(tid<s) {
        if(result_s[tid+s].x*result_s[tid+s].x+result_s[tid+s].y*result_s[tid+s].y >
	        result_s[tid].x*result_s[tid].x+result_s[tid].y*result_s[tid].y) {
				
	  result_s[tid].x = result_s[tid+s].x;
	  result_s[tid].y = result_s[tid+s].y;	
	
	}  
      }
      __syncthreads();
    }
    
    if(tid==0) {
      result[blockIdx.x].x = result_s[0].x; 
      result[blockIdx.x].y = result_s[0].y;  
    }   
  }
}

__global__ void linstep(cuComplex* fNew, cuComplex* fOld, float dt)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  cuComplex cj;
  cj.x = 0.0f;
  cj.y = 1.0f;
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;

		fNew[index] = fOld[index] * exp(cj*kz(idz)*dt);

    }
  }
  
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
		int IDZ = idz + zThreads*i;

		fNew[index] = fOld[index] * exp(cj*kz(IDZ)*dt);
	
      }
    }
  }
}


__global__ void fwdeuler(cuComplex* fNew, cuComplex* nl, float dt)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;

		fNew[index] = fNew[index] + nl[index]*dt;

    }
  }
  
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;

		fNew[index] = fNew[index] + nl[index]*dt;
       }
     }
    }     
}

__global__ void fwdeuler(float* fNew, float* nl, float dt)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;

		fNew[index] = fNew[index] + nl[index]*dt;

    }
  }
  
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;

		fNew[index] = fNew[index] + nl[index]*dt;
       }
     }
    }     
}


// Multiply array by kperp**2
__global__ void multKPerp(cuComplex* fK, cuComplex* f, float scaler)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  // kPerp2 is defined with a minus sign, kperp2 = -( kx**2 + ky**2)
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
        fK[index] = f[index] * kPerp2(idx, idy) * scaler;
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
          fK[index] = f[index] * kPerp2(idx,idy) * scaler;
        
      }
    }
  }
}       
__global__ void multKPerpInv(cuComplex* fK, cuComplex* f)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  // kPerp2 is defined with a minus sign, kperp2 = -( kx**2 + ky**2)
  // But that's alright, because you want to divide through by nabla^2 = -kperp^2
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
        fK[index] = .5f * f[index] * kPerp2Inv(idx,idy);
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
          fK[index] = .5f * f[index] * kPerp2Inv(idx,idy);
        
      }
    }
  }
}       

// Multiply array by kx
__global__ void multKx(cuComplex* fK, cuComplex* f) 
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
      fK[index] = f[index]*kx(idx);
      		 
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
        fK[index] = f[index]*kx(idx);
        
      }
    }
  }
}   

// Multiply array by ky
__global__ void multKy(cuComplex* fK, cuComplex* f) 
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
      fK[index].x = f[index].x * ky(idy);
      fK[index].y = f[index].y * ky(idy);
      		 
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
        fK[index].x = f[index].x * ky(idy);
        fK[index].y = f[index].y * ky(idy);
        
      }
    }
  }
}           

// Add, subtract arrays
__global__ void addsubt(cuComplex* result, cuComplex* f, cuComplex* g, float a)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
        result[index] = f[index] + a*g[index];
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
        result[index] = f[index] + a*g[index];
      }
    }
  }
}           
// Multiply arrays C = A*B
__global__ void mult(float* C, float* A, float* B)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads){
  	if(idy<Ny && idx<Nx && idz<Nz){
		int index = idy + Ny*idx + Nx*Ny*idz;
		C[index] = A[index]*B[index];
		}
	}
  
  else{
  	for(int i=0; i<Nz/zThreads; i++){
		if(idy<Ny && idx<Nx && idz<zThreads){
			int index = idy + Ny*idx + Nx*Ny*idz + Nx*Ny*zThreads*i;
			C[index] = A[index]*B[index];
			}
		}
	}
}
// Square root
__global__ void squareroot(float* A)
{
  unsigned int idx = get_idx();
  A[idx] = sqrt(A[idx]);
}
// Divide arrays : C[index] = A[index]/B[index] if B[index]!=0
__global__ void divide(float* C, float* A, float* B)
{
  unsigned int index = get_idx();
  if(abs(B[index]) > 1.e-8) C[index] = A[index]/B[index];
  else C[index] = 0.0f;
}
// Square a real array and save it in f[].x
__global__ void square(float* f)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<Ny && idx<Nx && idz<Nz) {
      unsigned int index = idy + Ny*idx + Nx*Ny*idz;
      
      f[index] = f[index]*f[index];
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<Ny && idx<Nx && idz<zThreads) {
        unsigned int index = idy + Ny*idx + Nx*Ny*idz + Nx*Ny*zThreads*i;
	
        f[index] = f[index]*f[index];
      }
    }
  }
}    
// Square a complex array and save it in f[].x
__global__ void squareComplex(cuComplex* f)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
      f[index].x = f[index].x*f[index].x + f[index].y * f[index].y;
      f[index].y = 0;
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
        f[index].x = f[index].x*f[index].x + f[index].y * f[index].y;
		f[index].y = 0;
      }
    }
  }
}    
// Fix fft
__global__ void fixFFT(cuComplex* f)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
	if(idy!=0) f[index] = 2.0f*f[index];
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
	if(idy!=0) f[index] = 2.0f*f[index];
      }
    }

  }
}        	

////////////////////////////////////////
// Scale operations
////////////////////////////////////////
// Scale a complex array by a real number
__global__ void scale(cuComplex* b, float scaler)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + (Ny/2+1)*(Nx)*idz;
    
      b[index].x = scaler*b[index].x;
      b[index].y = scaler*b[index].y;
    }
  }
    
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
		b[index].x = scaler*b[index].x;
        b[index].y = scaler*b[index].y; 
      }
    }
  }    	
} 
// Scale a real array by a real number
__global__ void scale(float* b, float scaler)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + (Ny/2+1)*(Nx)*idz;
    
      b[index] = scaler*b[index];
    }
  }
    
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
		b[index] = scaler*b[index];
      }
    }
  }    	
} 
// Scale a complex array by a real number and save it in result
__global__ void scale(cuComplex* result, cuComplex* b, float scaler)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + (Ny/2+1)*(Nx)*idz;
    
      result[index].x = scaler*b[index].x;
      result[index].y = scaler*b[index].y;
    }
  }
    
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
		result[index].x = scaler*b[index].x;
        result[index].y = scaler*b[index].y; 
      }
    }
  }    	
} 

//copies f(ky[i]) into fky
__global__ void kycopy(cuComplex* fky, cuComplex* f, int i) {
    
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();

  if(idy<Nz && idx<Nx) {
    unsigned int index = idx + (Nx)*idy;
    fky[index].x = f[i + index*(Ny/2+1)].x;
    fky[index].y = f[i + index*(Ny/2+1)].y;
  } 
}      

/////////////////////////////////////////
// Zeroing out arrays
/////////////////////////////////////////
__global__ void zero(cuComplex* f, int nx, int ny, int nz) 
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(nz<=zThreads) {
   if(idy<ny && idx<nx && idz<nz) {
    unsigned int index = idx + nx*idy + nx*ny*idz;
    
    f[index].x = 0;
    f[index].y = 0;
   }
  }
  else {
   for(int i=0; i<nz/zThreads; i++) {
    if(idy<ny && idx<nx && idz<zThreads) {
    unsigned int index = idx + nx*idy + nx*ny*idz + nx*ny*zThreads*i;
    
    f[index].x = 0;
    f[index].y = 0;
    }
   }
  }    
}    

__global__ void zero(float* f, int nx, int ny, int nz) 
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(nz<=zThreads) {
   if(idy<ny && idx<nx && idz<nz) {
    unsigned int index = idx + nx*idy + nx*ny*idz;
    
    f[index] = 0;
   }
  }
  else {
   for(int i=0; i<nz/zThreads; i++) {
    if(idy<ny && idx<nx && idz<zThreads) {
    unsigned int index = idx + nx*idy + nx*ny*idz + nx*ny*zThreads*i;
    
    f[index] = 0;
    }
   }
  }    
}    

__global__ void zderiv(cuComplex* result, cuComplex* f)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  cuComplex res, cj;
  cj.x = 0.0f;
  cj.y = 1.0f;
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;

	  res = cj*kz(idz)*f[index];
	  result[index] = res;
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
		int IDZ = idz + zThreads*i;
		 res = cj*kz(IDZ)*f[index];
		 result[index] = res;
      }
    }
  }    	
} 

__global__ void absk_closure(cuComplex* f, float scaler)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
      //result(ky,kx,kz)= i*kz*f(ky,kx,kz)
      f[index].x = exp(abs(kz(idz))*scaler) * f[index].x;
      f[index].y = exp(abs(kz(idz))*scaler) * f[index].y;    
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
		int IDZ = idz + zThreads*i;
	
		f[index].x = exp(abs(kz(IDZ))*scaler) * f[index].x;
        f[index].y = exp(abs(kz(IDZ))*scaler) * f[index].y;  
      }
    }
  }    	
} 


