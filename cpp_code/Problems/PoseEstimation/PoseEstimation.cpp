
#include "Problems/PoseEstimation/PoseEstimation.h"

/*Define the namespace*/
namespace ROPTLIB{

	PoseEstimation::PoseEstimation(double *inA, double *inB, double *inD, integer inn, integer inp)
	{
	    A = inA;
		B = inB;
		D = inD;
		n = inn;
		p = inp;
		E = (double*)malloc(8*p* sizeof(double));
	};

	PoseEstimation::~PoseEstimation(void)
	{
	};

	double PoseEstimation::f(Variable *x) const
	{
		const double *xxM = x->ObtainReadData();
        double f;

        // E = deep copy of B
		for (int i = 0 ; i < 8; ++i)
		{
			for (int j = 0 ; j < p ; ++j)
			{
				E[i + j*8] = B[i + j*8];
			}
		}

		// temporary variables tmp1[p*8] = E'D and tmp2[p*p] = E'DE
		double* tmp1 = (double*)malloc(p*8* sizeof(double));
		double* tmp2 = (double*)malloc(p*p* sizeof(double));

        Matrix ME(E, 8, p), MxxM(xxM, n, p), MA(A, n, 8), MD(D, 8, 8), Mtmp1(tmp1, p, 8), Mtmp2(tmp1, p, p);


        // E = A'*X - B
        Matrix::DGEMM(1.0, MA, true, MxxM, false, -1.0, ME);

		// tmp1 = E'D, tmp2 = E'DE
        Matrix::DGEMM(1.0, ME, true, MD, false, 0.0, Mtmp1);
        Matrix::DGEMM(1.0, Mtmp1, false, ME, false, 0.0, Mtmp2);

		f = Mtmp2.matrix[0] + Mtmp2.matrix[3]; // f = trace(E'DE)
        return f;
	};

	void PoseEstimation::EucGrad(Variable *x, Vector *egf) const
	{
		PoseEstimation::f(x);
		double *xegf = egf->ObtainWriteEntireData();
		double *tmp1 = (double *) malloc(n * 8 * sizeof(double));
		double *tmp2 = (double *) malloc(8 * p * sizeof(double));
		Matrix ME(E, 8, p), MA(A, n, 8), MD(D, 8, 8), Mtmp1(tmp1, n, 8), Mtmp2(tmp2, n, p), Megf(xegf, n, p);

		Matrix::DGEMM(1.0, MA, false, MD, false, 0.0, Mtmp1);
		Matrix::DGEMM(1.0, Mtmp1, false, ME, false, 0.0, Mtmp2);
		Matrix::DGEMM(1.0, Mtmp1, false, ME, false, 0.0, Megf);

	};
}; /*end of ROPTLIB namespace*/

