// Make this more like Viriato: Input kp1 and kp2 from input file. Then calculate all the
// kperps in between, and force them.
// 
// BUT, we will drive all members of this set of modes at each time step
//
void forcing( cuComplex *force, float dt, int *kstir_x, int *kstir_y, int *kstir_z, float fampl)
{
    float phase,amp,kp;
    cuComplex *temp;
    unsigned int index;
    temp = (cuComplex*) malloc(sizeof(cuComplex));

    int id = rand() % nkstir;
    // replace this with dt_interval = dt;
    float dt_interval = dt * nkstir;
    // replace this kp = line with a loop over kstir_x, kstir_y and kstir_z
    kp = sqrt(-kPerp2(kstir_x[id], kstir_y[id]));

    float ran_amp = ( (float) rand()) / ((float) RAND_MAX + 1.0 );
    amp = (1.0/abs(kp)) * sqrt(abs((fampl/dt_interval)*log(ran_amp)));
    phase = M_PI*(2.0*( (float) rand()) / ((float) RAND_MAX + 1.0 ) -1.0);
    DEBUGPRINT("dt = %f, ran_amp = %f, amp = %f, phase = %f\n",dt, ran_amp, amp, phase);

    temp[0].x = amp*cos(phase);
    temp[0].y = amp*sin(phase);

    index = kstir_y[id] + (Ny/2+1)*kstir_x[id] +Nx*(Ny/2+1)*kstir_z[id];

    CP_TO_GPU(force + index, temp, sizeof(cuComplex));
    DEBUGPRINT("Copying forcing term to GPU : %s, id = %d, index = %d\n",
            cudaGetErrorString(cudaGetLastError()), id, index);


    // Reality condition
    // NEED: check whether this should be applied to kz=0 only
    if(kstir_y[id] == 0){

        temp[0].y = -temp[0].y;
        index = kstir_y[id] + (Ny/2+1)*((Nx-kstir_x[id])%Nx) + Nx*(Ny/2+1)*((Nz-kstir_z[id])%Nz);
        CP_TO_GPU(force + index, temp, sizeof(cuComplex));
        CUDA_DEBUG("Copying complex conjugate of forcing term to GPU : %s\n"); 
    }

    free(temp);

    CUDA_DEBUG("Exiting forcing : %s\n"); 

}

// This is not coded correctly
void force_ant( cuComplex *force, float dt, int *kstir_x, int *kstir_y, int *kstir_z, float fampl)
{
    float phase,amp,kp;
    cuComplex *temp;
    unsigned int index;
    temp = (cuComplex*) malloc(sizeof(cuComplex));

    int id = rand() % nkstir;

    kp = sqrt(-kPerp2(kstir_x[id], kstir_y[id]));

    float ran_amp = ( (float) rand()) / ((float) RAND_MAX + 1.0 );

    phase = M_PI*(2.0*( (float) rand()) / ((float) RAND_MAX + 1.0 ) -1.0);
    DEBUGPRINT("dt = %f, ran_amp = %f, amp = %f, phase = %f\n",dt, ran_amp, amp, phase);

    temp[0].x = amp*cos(phase);
    temp[0].y = amp*sin(phase);

    index = kstir_y[id] + (Ny/2+1)*kstir_x[id] +Nx*(Ny/2+1)*kstir_z[id];
    CP_TO_GPU(force + index, temp, sizeof(cuComplex));

    DEBUGPRINT("Copying forcing to GPU : %s, id = %d, index = %d\n",
            cudaGetErrorString(cudaGetLastError()), id, index);

    // Reality condition
    if(kstir_y[id] == 0){

        temp[0].y = -temp[0].y;
        index = kstir_y[id] + (Ny/2+1)*((Nx-kstir_x[id])%Nx) + Nx*(Ny/2+1)*((Nz-kstir_z[id])%Nz);
        CP_TO_GPU(force + index, temp, sizeof(cuComplex));

        DEBUGPRINT("Copying complex conjugate of forcing term to GPU : %s\n",
                cudaGetErrorString(cudaGetLastError()));

    }

    free(temp);

    CUDA_DEBUG("Exiting forcing : %s\n");

}
