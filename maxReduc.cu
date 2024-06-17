void maxReduc(cuComplex* max, cuComplex* f, cuComplex* padded)
{

    zero <<<dG, dB>>> (padded,Nx,Ny,Nz);
    CP_ON_GPU(padded, f, Nkc);

    dim3 dBReduc(8,8,8);
    int gridx = (Nx*Ny*Nz)/512;

    if (Nx*Ny*Nz <= 512) {
        dBReduc.x = Nx;
        dBReduc.y = Ny;
        dBReduc.z = Nz;
        gridx = 1;
    }

    dim3 dGReduc(gridx,1,1);

    maximum <<<dGReduc,dBReduc,sizeof(cuComplex)*8*8*8>>> (padded, padded);

    while(dGReduc.x > 512) {
        dGReduc.x = dGReduc.x / 512;
        maximum <<<dGReduc,dBReduc,sizeof(cuComplex)*8*8*8>>> (padded, padded);
    }

    dBReduc.x = dGReduc.x;
    dGReduc.x = 1;
    dBReduc.y = dBReduc.z = 1;
    maximum <<<dGReduc,dBReduc,sizeof(cuComplex)*8*8*8>>> (padded,padded);

    CP_TO_CPU(max, padded, sizeof(cuComplex));
}

void sumReduc(cuComplex* result, cuComplex* f, cuComplex* padded)
{
    zero <<<dG, dB>>> (padded, Nx, Ny, Nz);
    CP_ON_GPU(padded, f, Nkc);

    dim3 dBReduc(8,8,8);
    int gridx = (Nx*Ny*Nz)/512;

    if (Nx*Ny*Nz <= 512) {
        dBReduc.x = Nx;
        dBReduc.y = Ny;
        dBReduc.z = Nz;
        gridx = 1;
    }

    dim3 dGReduc(gridx,1,1);

    sum <<<dGReduc,dBReduc,sizeof(cuComplex)*8*8*8>>> (padded, padded);

    while(dGReduc.x > 512) {
        dGReduc.x = dGReduc.x / 512;
        sum <<<dGReduc,dBReduc,sizeof(cuComplex)*8*8*8>>> (padded, padded);
    }

    dBReduc.x = dGReduc.x;
    dGReduc.x = 1;
    dBReduc.y = dBReduc.z = 1;
    sum <<<dGReduc,dBReduc,sizeof(cuComplex)*8*8*8>>> (padded,padded);

    CP_TO_CPU(result, padded, sizeof(cuComplex));

}

void sumReduc(float* result, float* f, float* padded)
{
    zero <<<dG, dB>>> (padded, Nx, Ny, Nz);
    CP_ON_GPU(padded, f, Nkf);

    dim3 dBReduc(8,8,8);
    int gridx = (Nx*Ny*Nz)/512;

    if (Nx*Ny*Nz <= 512) {
        dBReduc.x = Nx;
        dBReduc.y = Ny;
        dBReduc.z = Nz;
        gridx = 1;
    }

    dim3 dGReduc(gridx,1,1);

    sum <<<dGReduc,dBReduc,sizeof(float)*8*8*8>>> (padded, padded);

    while(dGReduc.x > 512) {
        dGReduc.x = dGReduc.x / 512;
        sum <<<dGReduc,dBReduc,sizeof(float)*8*8*8>>> (padded, padded);
    }

    dBReduc.x = dGReduc.x;
    dGReduc.x = 1;
    dBReduc.y = dBReduc.z = 1;
    sum <<<dGReduc,dBReduc,sizeof(float)*8*8*8>>> (padded,padded);

    CP_TO_CPU(result, padded, sizeof(float));
}


////////////////////////////////////////
void sumReduc_kz(cuComplex* result, cuComplex* f, cuComplex* padded)
{
    zero <<<dG, dB>>> (padded,Nx,Ny,1);
    CP_ON_GPU(padded, f, sizeof(cuComplex)*Nx*(Ny/2+1));

    dim3 dBReduc(8,8,8);
    int gridx = (Nx*Ny)/512;

    if (Nx*Ny <= 512) {
        dBReduc.x = Nx;
        dBReduc.y = Ny;
        dBReduc.z = 1;
        gridx = 1;
    }

    dim3 dGReduc(gridx,1,1);

    sum <<<dGReduc,dBReduc,sizeof(cuComplex)*8*8*8>>> (padded, padded);

    while(dGReduc.x > 512) {
        dGReduc.x = dGReduc.x / 512;
        sum <<<dGReduc,dBReduc,sizeof(cuComplex)*8*8*8>>> (padded, padded);
    }

    dBReduc.x = dGReduc.x;
    dGReduc.x = 1;
    dBReduc.y = dBReduc.z = 1;
    sum <<<dGReduc,dBReduc,sizeof(cuComplex)*8*8*8>>> (padded,padded);

    CP_TO_CPU(result, padded, sizeof(cuComplex));
}

////////////////////////////////////////
void sumReduc_gen(cuComplex* result, cuComplex* f, cuComplex* padded, int nx, int ny, int nz)
{
    zero <<<dG, dB>>> (padded, nx, ny, nz);
    DEBUGPRINT("nx = %d, ny = %d, nz = %d \n", nx, ny, nz);
    DEBUGPRINT("dG = %d, %d, %d \t dB = %d, %d, %d \n", dG.x, dG.y, dG.z, dB.x, dB.y, dB.z);
    CUDA_DEBUG("zero padded in sumreduc: %s\n");

    CP_ON_GPU(padded, f, sizeof(cuComplex)*nx*(ny/2+1)*nz);

    dim3 dBReduc(8,8,8);
    int gridx = (nx*ny*nz)/512;

    if (nx*ny*nz <= 512) {
        dBReduc.x = nx;
        dBReduc.y = ny;
        dBReduc.z = nz;
        gridx = 1;
    }

    dim3 dGReduc(gridx,1,1);

    sum <<<dGReduc,dBReduc,sizeof(cuComplex)*8*8*8>>> (padded, padded);

    while(dGReduc.x > 512) {
        dGReduc.x = dGReduc.x / 512;
        sum <<<dGReduc,dBReduc,sizeof(cuComplex)*8*8*8>>> (padded, padded);
    }

    dBReduc.x = dGReduc.x;
    dGReduc.x = 1;
    dBReduc.y = dBReduc.z = 1;
    sum <<<dGReduc,dBReduc,sizeof(cuComplex)*8*8*8>>> (padded,padded);

    CP_TO_CPU(result, padded, sizeof(cuComplex));
}

void sumReduc_gen(float* result, float* f, float* padded, int nx, int ny, int nz)
{
    zero <<<dG, dB>>> (padded, nx, ny, nz);
    CP_ON_GPU(padded,f,sizeof(float)*nx*(ny/2+1)*nz);

    dim3 dBReduc(8,8,8);
    int gridx = (nx*ny*nz)/512;

    if (nx*ny*nz <= 512) {
        dBReduc.x = nx;
        dBReduc.y = ny;
        dBReduc.z = nz;
        gridx = 1;
    }

    dim3 dGReduc(gridx,1,1);

    sum <<<dGReduc,dBReduc,sizeof(float)*8*8*8>>> (padded, padded);

    while(dGReduc.x > 512) {
        dGReduc.x = dGReduc.x / 512;
        sum <<<dGReduc,dBReduc,sizeof(float)*8*8*8>>> (padded, padded);
    }

    dBReduc.x = dGReduc.x;
    dGReduc.x = 1;
    dBReduc.y = dBReduc.z = 1;
    sum <<<dGReduc,dBReduc,sizeof(float)*8*8*8>>> (padded,padded);

    CP_TO_CPU(result, padded, sizeof(float));
}

void sumReduc_gen(float* result, float* f, int nx, int ny, int nz)
{

    dim3 dBReduc(8,8,8);
    int gridx = (nx*ny*nz)/512;

    if (nx*ny*nz <= 512) {
        dBReduc.x = nx;
        dBReduc.y = ny;
        dBReduc.z = nz;
        gridx = 1;
    }

    dim3 dGReduc(gridx,1,1);

    sum <<<dGReduc,dBReduc,sizeof(float)*8*8*8>>> (f, f);

    while(dGReduc.x > 512) {
        dGReduc.x = dGReduc.x / 512;
        sum <<<dGReduc,dBReduc,sizeof(float)*8*8*8>>> (f, f);
    }

    dBReduc.x = dGReduc.x;
    dGReduc.x = 1;
    dBReduc.y = dBReduc.z = 1;
    sum <<<dGReduc,dBReduc,sizeof(float)*8*8*8>>> (f,f);

    CP_TO_CPU(result, f, sizeof(float));
}
