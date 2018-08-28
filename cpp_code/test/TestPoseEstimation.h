//
// Created by rnb on 18. 8. 20.
//

/*
This is the test file for the Brocokett problem defined in StieBrockett.h and StieBrockett.cpp.

---- WH
*/

#ifndef TESTSTIEBROCKETT_H
#define TESTSTIEBROCKETT_H

/*Output to console*/
#include <iostream>

/*Generate random number*/
#include "Others/randgen.h"

/*Computational time*/
#include <ctime>

/*Problem related classes*/
#include "Problems/Problem.h"
#include "Problems/PoseEstimation/PoseEstimation.h"
#include "Problems/PoseEstimation/utils.h"

/*Manifold related classes*/
#include "Manifolds/Manifold.h"
#include "Manifolds/Stiefel/StieVector.h"
#include "Manifolds/Stiefel/StieVariable.h"
#include "Manifolds/Stiefel/Stiefel.h"

/*Trust-region based solvers*/
#include "Solvers/SolversTR.h"
#include "Solvers/RTRSR1.h"


/*The global head file*/
#include "Others/def.h"

/*The general headers*/
#include <pwd.h>
#include <stdio.h>
#include <iostream>
#include <signal.h>
#include "../utils_struct2.h"

/*OpenCV*/
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/*GSL*/
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>

using namespace ROPTLIB;
using namespace cv;
using namespace std;

#if defined(TESTPOSEESTIMATION)
int main(int argc,char* argv[]);
#endif


// create the pascal template, see the matlab code
int get_pascal_template(model cad, dictonnaire dict);

// this function implements the function prox_2norm in the matlab code, Z and lam are arguments and X and normX are the output
// Z is a m by n matrix m > n
// (I do not understand what it does, see the matlab code)
int prox_2norm(gsl_matrix* Z, double lam, gsl_matrix* X, double normX);

// this function implementes proj_deformable_approx in the matlab code (I do not understand what it does, see the matlab code)
// X is the input and Y,L and Q are the outputs
int proj_deformable_approx(gsl_matrix* X, gsl_matrix* Y, gsl_matrix* L, gsl_matrix* Q);

// this function returns the kronecker product between a(m*p) and b(n*q), the returned matrix will be m*n by p*q
gsl_matrix* matrix_KPro(gsl_matrix* a, gsl_matrix* b);

// this function implements the function syncRot (I do not understand what it does, see the matlab code)
// T(2*3) is the input and R(3*3) and C(double) are the outputs
int syncRot(gsl_matrix* T, gsl_matrix* R, gsl_matrix* C);

// this functions implements the function estimateR_weighted which find the optimal R with manifold optimization
// S, W2fit, D and R0 are the inputs, R is the output
gsl_matrix* estimateR_weighted(gsl_matrix* S, gsl_matrix* W, gsl_matrix* D, gsl_matrix* R0, bool verb);

// this functions implements the function estimateC_weighted which find the optimal C
// S, W2fit, D, R and lam are the inputs, C is the output
double estimateC_weighted(gsl_matrix* S, gsl_matrix* W, gsl_matrix* inD, gsl_matrix* R, double lam);

// this function implements PoseFromKps_WP (see matlab code)
pose pose_from_kps_WP(gsl_matrix* W, dictonnaire dict, gsl_vector* weight, bool verb, int lam_val, double tol);

// this function implements Pose FromKps_FP
pose pose_from_kps_FP(gsl_matrix* W, dictonnaire dict, gsl_matrix* R, gsl_vector* weight, bool verb, int lam_val, double tol);

// this function takes an image as input and return the key points
pose find_maximal_response(char imname[], char keypointname[], gsl_vector* defscale, gsl_matrix* defkp);

/*function to use the manifold optimization*/
gsl_matrix* testPoseEstimation(double *A, double *B, double *D, integer n, integer p, double *X = nullptr, double *Xopt = nullptr, bool verb=true);


#endif // end of TESTPOSEESTIMATION_H
