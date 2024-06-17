void fft_plan_create()
{
    if(cufftPlan3d(&plan_C2R, Nz, Nx, Ny, CUFFT_C2R) != CUFFT_SUCCESS) {
        printf("plan_C2R creation failed. Don't trust results. \n");
    };
    if(cufftPlan3d(&plan_R2C, Nz, Nx, Ny, CUFFT_R2C) != CUFFT_SUCCESS) {
        printf("plan_R2C creation failed. Don't trust results. \n");
    }
}

void fft_plan_destroy()
{
    if(cufftDestroy(plan_C2R) != CUFFT_SUCCESS) printf("plan_C2R destruction failed. \n");
    if(cufftDestroy(plan_R2C) != CUFFT_SUCCESS) printf("plan_R2C destruction failed. \n");
}

void NLPS(cuComplex *result, cuComplex *f, cuComplex *g)
{
    GRADIENT (f, dx, dy);

    if(cufftExecC2R(plan_C2R, dy, fdyR) != CUFFT_SUCCESS) printf("fdyR calculation failed. \n");	
    if(cufftExecC2R(plan_C2R, dx, fdxR) != CUFFT_SUCCESS) printf("fdxR calculation failed. \n");

    GRADIENT (g, dx, dy);

    if(cufftExecC2R(plan_C2R, dy, gdyR) != CUFFT_SUCCESS) printf("gdyR calculation failed.  \n");
    if(cufftExecC2R(plan_C2R, dx, gdxR) != CUFFT_SUCCESS) printf("gdyR calculation failed.  \n");

    // Reuse fdxR as result 
    bracket <<<dG,dB>>> (fdxR, fdxR, fdyR, gdxR, gdyR, 1.0);

    if(cufftExecR2C(plan_R2C, fdxR, result) != CUFFT_SUCCESS) printf("R2C failed. \n");  

    scale <<<dG,dB>>> (result,1.0f/((float) Nx*Ny*Nz));

    // Dealias
    mask <<<dG,dB>>> (result);
}

