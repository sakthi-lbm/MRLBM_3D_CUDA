#ifndef RECONSTRUCTION_CUH
#define RECONSTRUCTION_CUH

#include "config.h"
#include "all_headers.h"

__device__ inline void pop_reconstruction(const real rho,
										  const real ux, const real uy, const real uz,
										  const real mxx, const real myy, const real mzz,
										  const real mxy, const real mxz, const real myz,
										  real *pop)
{
	real multiplyTerm = rho * W0;
	const real pics2 = toReal(1.0) - cs2 * (mxx + myy + mzz);

	pop[0] = multiplyTerm * (pics2);

	multiplyTerm = rho * W1;
	pop[1] = multiplyTerm * (pics2 + ux + mxx);
	pop[2] = multiplyTerm * (pics2 - ux + mxx);
	pop[3] = multiplyTerm * (pics2 + uy + myy);
	pop[4] = multiplyTerm * (pics2 - uy + myy);
	pop[5] = multiplyTerm * (pics2 + uz + mzz);
	pop[6] = multiplyTerm * (pics2 - uz + mzz);

	multiplyTerm = rho * W2;
	pop[7] = multiplyTerm * (pics2 + ux + uy + mxx + myy + mxy);
	pop[8] = multiplyTerm * (pics2 - ux - uy + mxx + myy + mxy);
	pop[9] = multiplyTerm * (pics2 + ux + uz + mxx + mzz + mxz);
	pop[10] = multiplyTerm * (pics2 - ux - uz + mxx + mzz + mxz);
	pop[11] = multiplyTerm * (pics2 + uy + uz + myy + mzz + myz);
	pop[12] = multiplyTerm * (pics2 - uy - uz + myy + mzz + myz);
	pop[13] = multiplyTerm * (pics2 + ux - uy + mxx + myy - mxy);
	pop[14] = multiplyTerm * (pics2 - ux + uy + mxx + myy - mxy);
	pop[15] = multiplyTerm * (pics2 + ux - uz + mxx + mzz - mxz);
	pop[16] = multiplyTerm * (pics2 - ux + uz + mxx + mzz - mxz);
	pop[17] = multiplyTerm * (pics2 + uy - uz + myy + mzz - myz);
	pop[18] = multiplyTerm * (pics2 - uy + uz + myy + mzz - myz);

	multiplyTerm = rho * W3;
	pop[19] = multiplyTerm * (pics2 + ux + uy + uz + mxx + myy + mzz + (mxy + mxz + myz));
	pop[20] = multiplyTerm * (pics2 - ux - uy - uz + mxx + myy + mzz + (mxy + mxz + myz));
	pop[21] = multiplyTerm * (pics2 + ux + uy - uz + mxx + myy + mzz + (mxy - mxz - myz));
	pop[22] = multiplyTerm * (pics2 - ux - uy + uz + mxx + myy + mzz + (mxy - mxz - myz));
	pop[23] = multiplyTerm * (pics2 + ux - uy + uz + mxx + myy + mzz - (mxy - mxz + myz));
	pop[24] = multiplyTerm * (pics2 - ux + uy - uz + mxx + myy + mzz - (mxy - mxz + myz));
	pop[25] = multiplyTerm * (pics2 - ux + uy + uz + mxx + myy + mzz - (mxy + mxz - myz));
	pop[26] = multiplyTerm * (pics2 + ux - uy - uz + mxx + myy + mzz - (mxy + mxz - myz));
}

#endif // !RECONSTRUCTION_CUH
