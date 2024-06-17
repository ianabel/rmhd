// Nonlinear timestepping routine using Noah's trick
void alf_adv(cuComplex *zpNew, 
        cuComplex *zpOld, 
        cuComplex *zpstar, 
        cuComplex *zmNew, 
        cuComplex *zmOld, 
        cuComplex *zmstar, 
        float dt) {

    if (nlrun) {
        // temp1 = {zp, -kperp**2 *zm} + {zm, -kperp**2 zp}, temp2 = -kperp**2{zp,zm}
        nonlin(zpstar, zmstar, temp1, temp2, temp3);

        //ZP
        // temp3 = temp1 - temp2 for zp
        //    addsubt <<<dG,dB>>> (temp3, temp1, temp2, -1);
        ADDSUBT (temp3, temp1, temp2, -1);
        // Coeff of 0.5 in front of the nonlinear term is included in multkperpinv
        // multiply by kperp2**(-1)
        multKPerpInv <<<dG,dB>>> (temp3, temp3);

        // multiply nonlinear term by integrating factor
        linstep <<<dG,dB>>> (temp3, temp3, dt);

    }
    // zpNew = zpOld*exp(i*kz*dt)
    //  linstep <<<dG,dB>>> (zpNew, zpOld, dt);
    LINSTEP (zpNew, zpOld, dt);

    if (nlrun) {
        // Add in the nonlinear term
        fwdeuler <<<dG,dB>>> (zpNew, temp3, dt);
    }

    if (nlrun) {
        //ZM
        // temp3 = bracket1 + bracket2 for zm
        addsubt <<<dG,dB>>> (temp3, temp1, temp2, 1);

        // Coeff of .5 in front of the nonlinear term is included in multkperpinv
        // multiply by kperp2**(-1)
        multKPerpInv <<<dG,dB>>> (temp3, temp3);

        // multiply nonlinear term by integrating factor
        LINSTEP(temp3, temp3, -dt);
    }

    // zmNew = zmOld*exp(-i*kz*dt)
    LINSTEP (zmNew, zmOld, -dt);

    if (nlrun) {
        // Add in the nonlinear term
        fwdeuler <<<dG,dB>>> (zmNew, temp3, dt);
    }
}

// Timestepping routine for alfven, nonlinear, nondebug
void advance(cuComplex *zpNew, cuComplex *zpOld, cuComplex *zmNew, cuComplex *zmOld, float dt, int istep) {

    if(driven && istep%nforce==0){

        // Alfven wave forcing
        zero <<<dG,dB>>> (temp1, Nx, Ny/2+1, Nz);
        forcing(temp1, dt, kstir_x, kstir_y, kstir_z, fampl);
        fwdeuler <<<dG,dB>>> (zpOld, temp1, dt);
        fwdeuler <<<dG,dB>>> (zmOld, temp1, dt);
    }

    // Half Alfven step
    alf_adv(zpNew, zpOld, zpOld, zmNew, zmOld, zmOld, dt/2.0);

    // Full Alfven step
    alf_adv(zpNew, zpOld, zpNew, zmNew, zmOld, zmNew, dt);

    // Damping
    damp_hyper <<<dG,dB>>> (zpNew, nu_hyper, alpha_hyper, dt);
    damp_hyper <<<dG,dB>>> (zmNew, nu_hyper, alpha_hyper, dt);

    dampz <<<dG,dB>>> (zpNew, nu_kz, alpha_z, dt);
    dampz <<<dG,dB>>> (zmNew, nu_kz, alpha_z, dt);

    // Move the results, the zNew's, to the zOld's 
    CP_ON_GPU(zpOld, zpNew, Nkc);
    CP_ON_GPU(zmOld, zmNew, Nkc);
}

