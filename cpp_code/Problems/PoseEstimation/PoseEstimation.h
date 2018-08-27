/*
This file defines the class for the problem


Problem --> PoseEstimation

---- WH
*/

#ifndef POSEESTIMATION_H
#define POSEESTIMATION_H

#include "Manifolds/Stiefel/Stiefel.h"
#include "Manifolds/Stiefel/StieVariable.h"
#include "Manifolds/Stiefel/StieVector.h"
#include "Problems/Problem.h"
#include "Manifolds/SharedSpace.h"
#include "Others/def.h"
#include "Others/MyMatrix.h"

/*GSL*/
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>

/*Utils*/
#include "Problems/PoseEstimation/utils.h"

/*Define the namespace*/
namespace ROPTLIB{

	class PoseEstimation : public Problem{
	public:
		PoseEstimation(double *inA, double *inB, double *inD, integer inn, integer inp);
		virtual ~PoseEstimation();
		virtual double f(Variable *x) const;

		//virtual void RieGrad(Variable *x, Vector *gf) const;
		//virtual void RieHessianEta(Variable *x, Vector *etax, Vector *xix) const;

		virtual void EucGrad(Variable *x, Vector *egf) const;
		//virtual void EucHessianEta(Variable *x, Vector *etax, Vector *exix) const;
		double *A;
		double *B;
		double *D;
		double* E;
		integer n;
		integer p;
	};
}; /*end of ROPTLIB namespace*/
#endif
