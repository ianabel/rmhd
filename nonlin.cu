#include "nlps.cu"
void nonlin( cuComplex *zpOld, cuComplex *zmOld, cuComplex *bracket1, cuComplex *bracket2, cuComplex* zK)
{
    //1) zK = -kPerp2*zmOld
    multKPerp <<<dG,dB>>> (zK, zmOld, 1);
    
    //2) bracket1 = {zp,-kperp2*zm}
    NLPS (bracket1, zpOld, zK);
    
    //3) zK = -kPerp2*zpOld
    multKPerp <<<dG,dB>>> (zK, zpOld,1);
    
    //4) bracket2 = {zm,-kPerp2*zp}
    NLPS (bracket2, zmOld, zK);
    
    //5) bracket1 = {zp,-kPerp2*zm}+{zm,-kPerp2*zp}
    addsubt <<<dG,dB>>> (bracket1, bracket1, bracket2, 1);  //result put in bracket1
    
    //6) bracket2 = {zp,zm}
    NLPS (bracket2, zpOld, zmOld);
    
    //7) bracket2 = -kPerp2*[{zp,zm}]
    multKPerp <<<dG,dB>>> (bracket2, bracket2,1);

}

