/*Here are written all the functions that are not really specific to the problem*/

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

/*Output to console*/
#include <iostream>

/*Generate random number*/
#include "Others/randgen.h"

/*Computational time*/
#include <ctime>

/*The general headers*/
#include <pwd.h>
#include <stdio.h>
#include <iostream>
#include <signal.h>

using namespace cv;
using namespace std;


#ifndef C_TRADUCTION_UTILS_H
#define C_TRADUCTION_UTILS_H

// this function prints a gsl matrix
int print_matrix(const gsl_matrix *mat);

// this function prints a gsl vector
int print_vector(const gsl_vector* vect);

// this function computes the sum of the following the specified axis
gsl_vector* matrix_sum(gsl_matrix* mat, uint8_t axis);

// this function returns the sum of all the element of the vector vect
double vector_sum(gsl_vector* vect);

// this function returns the inverse of mat, mat should be square and non singular
// for the algorithm Pose from Kpts, the singularity is not a issue because the probability of find a singular matrix between all the reel matrix is egal to zero
gsl_matrix* matrix_inversion(gsl_matrix* mat);

// this function computes the Frobenius norm of the matrix
double matrix_norm(gsl_matrix* A);

// this function returns the trace of A, A has to be square
double matrix_trace(gsl_matrix* A);

// this function normalizes a array 3*8
int normalizeS(gsl_matrix* res);

// this function computes the singular value decomposition of A(m*n), the output are U,V, S and S_vect
// A = USV' ; U : m*n (orthogonal) ; S : n*n (diagonal) ; V : n*n (orthogonal square) ; S_vect is just the vector version of S
int matrix_singular_value_decomposition(gsl_matrix* A, gsl_matrix* U, gsl_matrix* S, gsl_vector* S_vect,  gsl_matrix* V);

// this function takes an image as input and return the value and coordinate of the maximum
// res[0] : x , res[1] : y , res[2] : values
gsl_vector* find_max_image(Mat image);

// this function reads a cvs file and modifies the array data, it works, do not touch it !
//the file sould has 3 rows
int read_csv(char *filename, double data[][3]);

// this function returns the max of abs(A)
// if A = [-8,5,7,-4] the function will return -8
double matrix_max_abs(gsl_matrix* A);

// this function returns the sum of all the absolute value of the elements of A
double matrix_abs_sum(gsl_matrix* A);

// this function returns the cross product of u by v, u and v must have a lenght of 3
gsl_vector* vector_cross_product(const gsl_vector* u, const gsl_vector* v);

// this functions returns the dot product of u by v
double vector_dot_product(const gsl_vector* u, const gsl_vector* v);

// this function returns the inner product between A and B, A and B must have the same size
double matrix_inner_product(gsl_matrix* A, gsl_matrix* B);

// this function returns the pseudo inverse of A using the SVD
gsl_matrix* matrix_pinv(gsl_matrix* A);

// this function returns L : the Cholesky decomposition of the matrix A. A = LL'
gsl_matrix* matrix_cholesky(gsl_matrix* A);

// returns A*B
gsl_matrix* matrix_product(const gsl_matrix* A, const gsl_matrix* B);

// returns A'
gsl_matrix* matrix_transpose(const gsl_matrix* A);

// returns centralised A
gsl_matrix* matrix_centralize(gsl_matrix* A);

// returns the std of A following the axis axis
gsl_vector* matrix_std(const gsl_matrix* A, int axis);

// return the term - to - term squared elements
gsl_matrix* matrix_square_elements(const gsl_matrix* A);

// returns the diagonal matrix of v
gsl_matrix* vector_diag(const gsl_vector* v);

// returns the vector of the diagonal of m
gsl_vector* matrix_diag(const gsl_matrix* m);

// sign function
double sign(double a);

// returns det(A)
double matrix_det(gsl_matrix *A);

#endif //C_TRADUCTION_UTILS_H
