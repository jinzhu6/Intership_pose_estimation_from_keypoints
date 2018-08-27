//
// Created by rnb on 18. 8. 24.
//


#include "test/TestPoseEstimation.h"



using namespace ROPTLIB;
using namespace cv;
using namespace std;


/*Help to check the memory leakage problem. No necesary any more.*/
std::map<integer *, integer> *CheckMemoryDeleted;



// create the pascal template, see the matlab code
int get_pascal_template(model cad, dictonnaire dict)
{

    gsl_matrix* S = gsl_matrix_calloc(3,8);

    for (int i = 0 ; i < 8 ; i++)
    {
        gsl_matrix_set(S,0,i,cad.kp[i][0]);
        gsl_matrix_set(S,1,i,cad.kp[i][1]);
        gsl_matrix_set(S,2,i,cad.kp[i][2]);
    }

    //copy of S in mu
    for (int i = 0 ; i < 3 ; ++i)
    {
        for(int j = 0 ; j < 8 ; j++)
        {
            gsl_matrix_set(dict.mu,i,j,gsl_matrix_get(S,i,j));
        }
    }

    // normalisation of S
    normalizeS(S);

    // copy of S into B
    for (int i = 0 ; i < 3 ; ++i)
    {
        for(int j = 0 ; j < 8 ; j++)
        {
            gsl_matrix_set(dict.B,i,j,gsl_matrix_get(S,i,j));
        }
    }
    return 0;
}

// this function implements the function prox_2norm in the matlab code, Z and lam are arguments and X and normX are the output
// Z is a m by n matrix m > n
// (I do not understand what it does, see the matlab code)
int prox_2norm(gsl_matrix* Z, double lam, gsl_matrix* X, double normX)
{
    // size of the matrix X, for the correct computation, X should be a 3 by 2 matrix
    uint8_t m = Z->size1;
    uint8_t n = Z->size2;

    // SV decomposition
    gsl_matrix* U = gsl_matrix_alloc(m,n);
    gsl_matrix* W = gsl_matrix_alloc(n,n);
    gsl_vector* W_vect = gsl_vector_alloc(n);
    gsl_matrix* V = gsl_matrix_alloc(n,n);

    matrix_singular_value_decomposition(Z, U, W, W_vect, V); // Z = UWV'

    // sum of the singular values
    double sum_SV = vector_sum(W_vect);

    if(sum_SV <= lam)
    {
        gsl_vector_set_zero(W_vect);
    }
    else if((gsl_vector_get(W_vect,0) - gsl_vector_get(W_vect,1)) <= lam)
    {
        gsl_vector_set(W_vect, 0, (sum_SV - lam)/2);
        gsl_vector_set(W_vect, 1, gsl_vector_get(W_vect,0));
    }
    else
    {
        gsl_vector_set(W_vect, 0, gsl_vector_get(W_vect,0) - lam);
    }

    // modification of W following W_vect (W_vect = diag(W))
    for (int i = 0 ; i < n ; i++)
    {
        gsl_matrix_set(W, i, i, gsl_vector_get(W_vect,i));
    }

    // caluclation of X : the goal is X = X = U*W*V' and normX = W_vect(1);
    gsl_matrix* tmp = gsl_matrix_calloc(m,n); // temp matrix variable
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, U, W, 0.0, tmp); // tmp = U*W
    gsl_matrix_transpose(V); // V = V'
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tmp, V, 0.0, X); // X = U*W*V'
    normX = gsl_vector_get(W_vect,1); // normX = W_tect(1)

    return 0 ;
}

// this function implementes proj_deformable_approx in the matlab code (I do not understand what it does, see the matlab code)
// X is the input and Y,L and Q are the outputs
int proj_deformable_approx(gsl_matrix* X, gsl_matrix* Y, gsl_matrix* L, gsl_matrix* Q)
{
    /* Ref: A. Del Bue, J. Xavier, L. Agapito, and M. Paladini, "Bilinear"
     * "Factorization via Augmented Lagrange Multipliers (BALM)" ECCV 2010.
     * This program is free software; you can redistribute it and/or
     * modify it under the terms of the GNU General Public License
     * as published by the Free Software Foundation; version 2, June 1991
     *
     * USAGE: proj_deformable_approx(X,Y,L,Q)
     *
     * This function projects a generic matrix X of size 3*K x 2 where K is the
     * number of basis shapes into the matrix Y that satisfy the manifold
     * constraints. This projection is an approximation of the projector
     * introduced in: M. Paladini, A. Del Bue, S. M. s, M. Dodig, J. Xavier, and
     * L. Agapito, "Factorization for Non-Rigid and Articulated Structure using
     * Metric Projections" CVPR 2009. Check the BALM paper, Sec 5.1.
     *
     * INPUT
     *
     * X: the 3*K x 2 affine matrix
     *
     * OUTPUT
     *
     * Y: the 3*K x 2 with manifold constraints
     * L: vector K
     * Q: gsl_matrix 3*K by 2
     */

    const uint8_t r = X->size1;
    const uint8_t d = r/3;
    gsl_matrix* A = gsl_matrix_calloc(3,3); // A = zeros(3,3)
    gsl_matrix* Ai = gsl_matrix_calloc(3*d,2); // A = zeros(3,3)
    gsl_vector* v = gsl_vector_calloc(2); // used to store the row of X

    // for each iteration of the loop the goal is to compute A = A + Ai*Ai' with Ai = X(i*3:i*3+1,:)
    for (int i = 0 ; i < d ; i++)
    {
        for (int j = i*3 ; j < 3*(i+1) ; j++)
        {
            gsl_matrix_get_row(v, X, j);
            gsl_matrix_set_row(Ai, j, v); // Ai = X(i*3:i*3+1,:)
        }
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Ai, Ai, 1, A); // A = A + Ai*Ai'
    }

    // we have the make the SV decomposition of A [U,S,V] = svd(A);
    gsl_matrix* U = gsl_matrix_calloc(3,3);
    gsl_matrix* S = gsl_matrix_calloc(3,3);
    gsl_vector* S_vect = gsl_vector_calloc(3);
    gsl_matrix* V = gsl_matrix_calloc(3,3);
    matrix_singular_value_decomposition(A, U, S, S_vect, V); // A = U*S*V'

    // the goal here is Q = U(:,1:2);
    gsl_vector* tmp = gsl_vector_calloc(3);
    gsl_matrix_set_zero(Q);
    for(int i = 0 ; i < 2 ; i++)
    {
        gsl_matrix_get_col(tmp, U, i);
        gsl_matrix_set_col(Q, i, tmp);
    }

    // calculation of G
    gsl_matrix* G = gsl_matrix_calloc(2,2);
    gsl_matrix* Gi = gsl_matrix_calloc(2,1);
    gsl_matrix* Ti = gsl_matrix_calloc(2,2);
    for (int i = 0 ; i < d ; i++)
    {
        for (int j = i*3 ; j < 3*(i+1) ; j++)
        {
            gsl_matrix_get_row(v, X, j);
            gsl_matrix_set_row(Ai, j, v); // Ai = X(i*3:i*3+1,:)
        }
        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Q, Ai, 0, Ti); // Ti = Q'*Ai
        gsl_matrix_set(Gi, 0, 0, matrix_trace(Ti)); // gi = [ trace(Ti) ; 0 ]
        gsl_matrix_set(Gi, 1, 0, gsl_matrix_get(Ti,1,0) - gsl_matrix_get(Ti,0,1)); // gi = [ trace(Ti) ; Ti(2,1)-Ti(1,2) ]
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Gi, Gi, 1, G); // G = G + Gi*Gi'
    }

    // recalculation of G
    gsl_matrix* U1 = gsl_matrix_calloc(2,2);
    gsl_matrix* S1 = gsl_matrix_calloc(2,2);
    gsl_vector* S1_vect = gsl_vector_calloc(2);
    gsl_matrix* V1 = gsl_matrix_calloc(2,2);
    matrix_singular_value_decomposition(G, U1, S1, S1_vect, V1); // G = U1*S1*V1'

    gsl_matrix_set_zero(G);
    gsl_matrix_set_zero(Ai);
    gsl_matrix_set_zero(Ti);
    gsl_matrix_set_zero(Gi);
    for (int i = 0 ; i < d ; i++)
    {
        for (int j = i*3 ; j < 3*(i+1) ; j++)
        {
            gsl_matrix_get_row(v, X, j);
            gsl_matrix_set_row(Ai, j, v); // Ai = X(i*3:i*3+1,:)
        }
        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Q, Ai, 0.0, Ti); // Ti = Q'*Ai
        gsl_matrix_set(Gi, 0, 0, gsl_matrix_get(Ti,0,0) - gsl_matrix_get(Ti,1,1)); // gi = [ Ti(1,1)-Ti(2,2) ; 0 ]
        gsl_matrix_set(Gi, 1, 0, gsl_matrix_get(Ti,1,0) + gsl_matrix_get(Ti,0,1)); // gi = [ trace(Ti) ; Ti(2,1)-Ti(1,2) ]
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Gi, Gi, 1.0, G); // G = G + Gi*Gi'
    }

    // calculation of thinks I absolutly do not understand
    gsl_matrix* U2 = gsl_matrix_calloc(2,2);
    gsl_matrix* S2 = gsl_matrix_calloc(2,2);
    gsl_vector* S2_vect = gsl_vector_calloc(2);
    gsl_matrix* V2 = gsl_matrix_calloc(2,2);
    matrix_singular_value_decomposition(G, U2, S2, S2_vect, V2); // G = U2*S2*V2'

    // computation of R
    gsl_matrix* R = gsl_matrix_calloc(2,2);
    double u1 , u2;
    if(gsl_matrix_get(S1,0,0) > gsl_matrix_get(S2,0,0))
    {
        u1 = gsl_matrix_get(U1,0,0);
        u2 = gsl_matrix_get(U1,0,1);
        gsl_matrix_set(R,0,0,u1);
        gsl_matrix_set(R,1,0,-u2);
        gsl_matrix_set(R,0,1,u2);
        gsl_matrix_set(R,1,1,u1);
    }
    else
    {
        u1 = gsl_matrix_get(U2,0,0);
        u2 = gsl_matrix_get(U2,0,1);
        gsl_matrix_set(R,0,0,u1);
        gsl_matrix_set(R,1,0,u2);
        gsl_matrix_set(R,0,1,u2);
        gsl_matrix_set(R,1,1,-u1);
    }

    // final computation of Q : the goal is Q = Q*R
    gsl_matrix* tmp2 = gsl_matrix_calloc(3,2);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Q, R, 0.0, tmp2); // tmp2 = Q*R
    gsl_matrix_memcpy(Q,tmp2); // Q = Q*R

    // computation of L and Y
    gsl_matrix_set_zero(L);
    gsl_matrix_set_zero(Y);
    gsl_vector* col_Y = gsl_vector_calloc(3); // will be used to store the col a ti*Q and set the col of Y
    double ti;
    gsl_matrix* tmp3 = gsl_matrix_calloc(2,2);
    for (int i = 0 ; i < d ; i++)
    {
        for (int j = i*3 ; j < 3*(i+1) ; j++)
        {
            gsl_matrix_get_row(v, X, j);
            gsl_matrix_set_row(Ai, j, v); // Ai = X(i*3:i*3+1,:)
        }
        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Q, Ai, 0.0, tmp3); // tmp3 = Q'*Ai
        ti = 0.5*matrix_trace(tmp3); // ti = 0.5*trace(Q'*Ai);

        gsl_matrix_set(L,i,0,ti);
        // the goal is Y = [ Y ; ti*Q ];
        for (int j = 2*i ; j < 2*(i+1) ; j++)
        {
            gsl_matrix_get_col(col_Y, Q, j); // col_Y = Q(:,j)
            gsl_vector_scale(col_Y,ti); // col_Y = ti*Q(:,j)
            gsl_matrix_set_col(Y,j,col_Y); // Y(:,j) = ti*Q(:,j)
        }
    }
    return 0;
}


// this function implements the function syncRot (I do not understand what it does, see the matlab code)
// T(2*3) is the input and R(3*3) and C(double) are the outputs
int syncRot(gsl_matrix* T, gsl_matrix* R, gsl_matrix* C)
{
    gsl_matrix* T_tr = gsl_matrix_calloc(3,2); // store the transpose of T
    gsl_matrix_transpose_memcpy(T_tr, T);
    gsl_matrix* Y = gsl_matrix_calloc(3,2); // will not be used
    gsl_matrix* L = gsl_matrix_calloc(1,1); // will store the output L of proj_deformable_approx
    gsl_matrix* Q = gsl_matrix_calloc(3,2); // will store the output Q of proj_deformable_approx
    gsl_matrix* C_tr = gsl_matrix_calloc(C->size2,C->size1); // will store the transpose of C
    gsl_matrix* R_tr = gsl_matrix_calloc(R->size2,R->size1-1); // will store the transpose of R but without the last colomne


    proj_deformable_approx(T_tr, Y, L, Q);

    double max_L = matrix_max_abs(L); // max in abs of L
    int s; // the sign of the max in abs of L
    if(max_L < 0) s = -1;
    else if(max_L == 0) s = 0;
    else s = 1;

    // C = s*L'
    gsl_matrix_memcpy(C_tr,L); // C' = L
    gsl_matrix_transpose_memcpy(C,C_tr); // C = C'' = L'
    gsl_matrix_scale(C,s); // C = s*L'

    // R(1:2,:) = s*Q'
    gsl_matrix_memcpy(R_tr,Q); // R_tr = Q
    gsl_matrix_scale(R_tr,s); // R_tr = s*Q'

    gsl_vector* R1 = gsl_vector_calloc(3);
    gsl_matrix_get_col(R1, R_tr, 0);
    gsl_vector* R2 = gsl_vector_calloc(3);
    gsl_matrix_get_col(R2, R_tr, 1);
    gsl_vector* R3 = vector_cross_product(R1,R2); // R(3,:) = cross(R(1,:),R(2,:));

    gsl_matrix_set_row(R,0,R1);
    gsl_matrix_set_row(R,1,R2);
    gsl_matrix_set_row(R,2,R3);
}

gsl_matrix* testPoseEstimation(double *A, double *B, double *D, integer n, integer p, double *X, double *Xopt, bool verb)
{
	//// choose a random seed
	//unsigned tt = (unsigned)time(NULL);
	////tt = 0;
	//genrandseed(tt);
	StieVariable StieX(n, p);

	if (X == nullptr)
	{/*If X is not defined before, then obtain an initial iterate by taking the Q factor of qr decomposition*/
		StieX.RandInManifold();
	}
	else
	{/*Otherwise, using the input orthonormal matrix as the initial iterate*/
		double *StieXptr = StieX.ObtainWriteEntireData();
		for (integer i = 0; i < n * p; i++)
			StieXptr[i] = X[i];
	}

	// Define the manifold
	Stiefel Domain(n, p);
	//Domain.SetHasHHR(true); /*set whether the manifold uses the idea in [HGA2015, Section 4.3] or not*/

	// Define the Brockett problem
	PoseEstimation Prob(A, B, D, n, p);
	/*The domain of the problem is a Stiefel manifold*/
	Prob.SetDomain(&Domain);

	/*Output the parameters of the domain manifold*/
	//Domain.CheckParams();


    RTRSR1 *solver = new RTRSR1(&Prob, &StieX);

	if (verb)
	{
	    printf("Solver RTRSD\n");
	    solver->Debug = FINALRESULT;
	}
	else
	{
	    solver->Debug = NOOUTPUT;
	}

	solver->Max_Iteration = 20;
	//solver->CheckParams();
	solver->Run();


	const Variable* xopt = solver->GetXopt();
    const double* xxopt = xopt->ObtainReadData();

	gsl_matrix* Xresult = gsl_matrix_alloc(p,n);
	// deep copy of xopt
	for (int i = 0 ; i < n; ++i)
    {
    	for (int j = 0 ; j < p ; ++j)
    	{
    		gsl_matrix_set(Xresult, j, i, xxopt[i + j*n]);
    	}
    }

    return Xresult;



}

// this functions implements the function estimateR_weighted which find the optimal R with manifold optimization
// S, W2fit, D and R0 are the inputs, R is the output
gsl_matrix* estimateR_weighted(gsl_matrix* S, gsl_matrix* W, gsl_matrix* D, gsl_matrix* R0, bool verb)
{

    gsl_matrix* A = gsl_matrix_calloc(S->size2,S->size1);
    gsl_matrix_transpose_memcpy(A,S);
    gsl_matrix* B = gsl_matrix_calloc(W->size2,W->size1);
    gsl_matrix_transpose_memcpy(B,W);
    gsl_matrix* X0 = gsl_matrix_calloc(R0->size2,R0->size1);
    gsl_matrix_transpose_memcpy(X0,R0); // X0 = R0' (this is equivalente to X0 = R0(1:2,:)' is the matlab code)

    int n = 3;
    int p = 2;

    gsl_matrix* R = testPoseEstimation(A->data, B->data, D->data, n, p, R0->data, nullptr, verb);

    return R;

}

double estimateC_weighted(gsl_matrix* W, gsl_matrix* R, gsl_matrix* B, gsl_matrix* inD, double lam)
{

    int p = W->size1;
    int k = B->size1 / 3;
    gsl_vector* d = gsl_vector_alloc(inD->size1); // d = diag(D)

    for(int i = 0 ; i < inD->size1 ; i++)
    {
        gsl_vector_set(d, i, gsl_matrix_get(inD ,i ,i));
    }
    gsl_matrix* D = gsl_matrix_calloc(2*p,2*p);

    for(int i = 0 ; i < p ; i++)
    {
        gsl_matrix_set(D, 2*i ,2*i , gsl_vector_get(d ,i));
        gsl_matrix_set(D, 2*i+1 ,2*i+1 , gsl_vector_get(d ,i));
    }

    // next we work on the linear system y = X*C
    gsl_matrix* y = gsl_matrix_calloc(p * W->size2, 1); // y is just W flattened
    for(int i = 0 ; i < W->size2 ; i++)
    {
        for(int j = 0 ; j < p ; j++)
        {
            gsl_matrix_set(y, i+j*W->size2, 0, gsl_matrix_get(W ,j ,i));
        }
    }

    gsl_matrix* X = gsl_matrix_calloc(2*p,k); // each column is rotated Bi
    gsl_matrix* Bi = gsl_matrix_calloc(3, B->size2); // Bi = B[3*ik:3*(ik+1),:]
    gsl_matrix* RBi = gsl_matrix_alloc(R->size1, B->size2);
    for(int ik = 0 ; ik < k ; ik++)
    {
        for(int i = 3*ik ; i < 3*(ik+1) ; i++)
        {
            for(int j = 0 ; j < B->size2 ; j++)
            {
                gsl_matrix_set(Bi, i, j, gsl_matrix_get(B, i, j));
            }
        }
        // here Bi = B[3*ik:3*(ik+1),:]

        // Rbi = R * B[3*i:3*(i+1),:]
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, R, Bi, 0.0, RBi);

        // X[:,i] = RBi flattened
        for(int i = 0 ; i < RBi->size1 ; i++)
        {
            for(int j = 0 ; j < RBi->size2 ; j++)
            {
                gsl_matrix_set(X, i + j*RBi->size1, ik, gsl_matrix_get(RBi ,i ,j));
            }
        }
    }

    // we want to calculate C = pinv(X'*D*X+lam*eye(size(X,2)))*X'*D*y and then C = C'
    gsl_matrix* XD = gsl_matrix_alloc(1,16);
    gsl_matrix* XDX = gsl_matrix_alloc(1,1);
    gsl_matrix* XDy = gsl_matrix_alloc(1,1);
    gsl_matrix* eye = gsl_matrix_alloc(X->size2,X->size2);
    gsl_matrix_set_identity(eye);

    gsl_matrix_scale(eye,lam);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, D, 0.0, XD);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, XD, X, 0.0, XDX);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, XD, y, 0.0, XDy);

    gsl_matrix* pre_inv = gsl_matrix_calloc(1,1); // pre_inv = X'*D*X+lam*eye(size(X,2))
    gsl_matrix* inv = gsl_matrix_calloc(1,1); // inv = pinv(X'*D*X+lam*eye(size(X,2)))
    gsl_matrix_add(pre_inv,XDX); gsl_matrix_add(pre_inv,eye); // inv = X'*D*X+lam*eye(size(X,2))
    inv = matrix_pinv(pre_inv);

    gsl_matrix* result = gsl_matrix_alloc(1,1);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, inv, XDy, 0.0, result);

    return gsl_matrix_get(result,0,0);

}

//this function implements PoseFromKps_WP (see matlab code)
pose pose_from_kps_WP(gsl_matrix* W, dictonnaire dict, gsl_vector* weight, bool verb, int lam_val, double tol)
{
    /* return : the pose of the object
     * W : the matrix of the heatmap
     * dict : the structure created by getpascaltemplate
     * weight : the vector of weight (I do not understand what it does, see the matlab code) [set every values of this vector at 1]
     * verb : verbose option, if true the results will be print during the iterations
     * lam_val : (I do not understand what it does, see the matlab code) [1]
     * tol : (I do not understand what it does, see the matlab code) [1e-3]
     *
     * RQ : I do not understand this function, so I am really soory for this implemantation, good luck ;)
     */

    // initialization of the variables
    const static short m = 3; // dimension of the space
    const short nb_joint = dict.nb_joints; // number of joints of the model
    const gsl_matrix* pc = dict.pc;

    gsl_matrix* B = gsl_matrix_calloc(m,nb_joint);
    gsl_matrix_memcpy(B,dict.mu); //B = mu

    const short k = (B->size1)/3;
    const short alpha = 1;

    gsl_matrix* D = vector_diag(weight); // D = diag(weight)
    gsl_matrix* lam = gsl_matrix_calloc(nb_joint, nb_joint); // lam = diag(lam_val)
    gsl_matrix_set_identity(lam);
    gsl_matrix_scale(lam,lam_val);
    double sum_D = vector_sum(weight); // sum_D = sum(weight)

    // computation of the mean of B following the axis 2
    gsl_matrix* B2 = matrix_centralize(B);
    gsl_matrix_memcpy(B,B2);

    //initialization of others variables
    gsl_matrix* M = gsl_matrix_calloc(2,3*k);
    gsl_matrix* M_t = gsl_matrix_calloc(3*k,2); // will be used later to store the transposed of M
    gsl_matrix* C = gsl_matrix_calloc(1,k);

    // auxiliary variables for ADMM
    gsl_matrix* Y = gsl_matrix_calloc(2,3*k);
    gsl_matrix* Z = gsl_matrix_calloc(2,3*k);
    gsl_matrix* Z0 = gsl_matrix_calloc(2,3*k);
    const double epsilon = DBL_EPSILON;

    // computation of mean abs(W)
    double mean_W = 0;
    for (int i = 0; i < nb_joint; i++)
    {
        mean_W += abs(gsl_matrix_get(W,i,0)) + abs(gsl_matrix_get(W,i,1));
    }

    mean_W /= (2*nb_joint);
    double mu = 1/(mean_W+epsilon);


    //print_matrix(W);
    // pre computing, compute BBt = B*D*B'
    gsl_matrix* BBt = gsl_matrix_calloc(m,m);
    gsl_matrix* tmp_BBt = gsl_matrix_calloc(3,nb_joint); // used during the calculation

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, B, D, 0.0, tmp_BBt);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, tmp_BBt, B, 0.0, BBt);

    // many variables will be used during the iteration, their initialization is here
    gsl_vector* T; // Translation vector
    gsl_matrix* tmp = gsl_matrix_calloc(2,nb_joint); // a temporary matrix variable
    gsl_matrix* tmp1 = gsl_matrix_calloc(2,nb_joint); // a temporary matrix variable
    gsl_matrix* tmp2 = gsl_matrix_calloc(2,m); // a temporary matrix variable
    gsl_matrix* tmp3 = gsl_matrix_calloc(2,m); // a temporary matrix variable
    gsl_matrix* tmp4 = gsl_matrix_calloc(m,m); // matrix identity m*m
    gsl_matrix* tmp5; // a temporary matrix variable
    double tmp6; // a tempory double variable
    gsl_matrix* Im = gsl_matrix_calloc(m,m); // matrix identity m*m
    gsl_matrix_set_identity(Im);
    gsl_matrix* W_t = gsl_matrix_calloc(W->size2,W->size1); // W_t is the transposed matrix of W
    gsl_matrix_transpose_memcpy(W_t,W);
    gsl_matrix* W2fit = gsl_matrix_calloc(W->size1,W->size2);
    gsl_matrix* Q = gsl_matrix_calloc(2,m);
    gsl_matrix* Q_t = gsl_matrix_calloc(m,2); // will be used later to store the transposed of Q
    double Primres; // will be used later
    double Dualres; // will be used later

    // iterations
    for(int iter = 0 ; iter < 1000 ; iter++)
    {

        // update translation : the goal is T = sum((W-Z*B)*D,2)/(sum(diag(D)) +eps) and W2fit = W - T*ones(1,p)
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, Z, B, 0.0, tmp); // tmp = -Z*B
        gsl_matrix_add(tmp,W_t); // tmp = W-Z*B
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tmp, D, 0.0, tmp1); //tmp1 = (W-Z*B)*D
        T = matrix_sum(tmp1,2); // T = sum((W-Z*B),2)
        gsl_vector_scale(T, 1/(sum_D+epsilon)); // T = sum((W-Z*B),2)/(sum(diad(D))+eps)

        // this for compute W2fit = W - T*ones(1,p);
        for (int i = 0; i < nb_joint; ++i)
        {
            for(int j = 0 ; j < T->size ; j++)
            {
                gsl_matrix_set(W2fit,i,j,gsl_matrix_get(W,i,j) - gsl_vector_get(T,j));
            }
        }


        // update motion matrix Z : the goal is Z = (W2fit*D*B'+mu*M+Y)/(BBt+mu*I3)
        gsl_matrix_memcpy(Z0,Z); // Z0 = Z
        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, W2fit, D, 0.0, tmp); // tmp = W2fit*D
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, tmp, B, 0.0, tmp2); // tmp2 = W2fit*D*B'
        gsl_matrix_scale(M,mu); // M = mu*M
        gsl_matrix_add(tmp2,M); // tmp2 = W2fit*D*B' + mu*M
        gsl_matrix_add(tmp2,Y); // tmp2 = W2fit*D*B' + mu*M + Y
        gsl_matrix_scale(M,1.0/mu); // M = M
        gsl_matrix_memcpy(tmp4,BBt); // tmp4 = BBt
        gsl_matrix_scale(Im,mu); // Im = mu*Im
        gsl_matrix_add(tmp4, Im); // tmp4 = BBt + mu*Im
        gsl_matrix_scale(Im,1.0/mu); // Im = Im
        tmp5 = matrix_inversion(tmp4); // tmp5 = inv(BBt+mu*Im)
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tmp2, tmp5, 0.0, Z); // Z = (W2fit*D*B' + mu*M +Y) / (BBt + mu*Im)

        // update the motion matrix M : the goal is Q = Z - Y/mu
        gsl_matrix_memcpy(Q,Z); // Q = Z
        gsl_matrix_scale(Y,-1.0/mu); // Y = -Y/mu
        gsl_matrix_add(Q,Y); // Q = Z - Y/mu
        gsl_matrix_scale(Y,-mu); // Y = Y

        // update de motion matrix : the goal is [M( :,3*i-2:3*i),C(i)] = prox_2norm(Q(:,3*i-2:3*i),alpha/mu) for all i between 0 and k-1
        // but in this case k == 1 so the goal is M,C = prox_2norm(Q,alpha/mu)
        // Q is a m by n matrix but m < n, It is necessary to transpose Q, the result of prox_2norm remain the same
        gsl_matrix_transpose_memcpy(Q_t,Q); // Q_t = Q'
        gsl_matrix_transpose_memcpy(M_t,M); // M_t = M'
        prox_2norm(Q_t, alpha/mu, M_t, tmp6); // M_t,tmp6 = prox_2norm(Q,alpha/mu)
        gsl_matrix_set(C, 0, 0, tmp6); // M_t,C = prox_2norm(Q,alpha/mu)
        gsl_matrix_transpose_memcpy(M,M_t); // M,C = prox_2norm(Q,alpha/mu)

        // update dual variable : the goal is Y = Y + mu*(M-Z)
        gsl_matrix_memcpy(tmp2, M); // tmp2 = M
        gsl_matrix_sub(tmp2, Z); // tmp2 = M-Z
        Primres = matrix_norm(tmp2); // we use the fact that tmp2 = M-Z to calculate norm(M-Z)
        gsl_matrix_scale(tmp2, mu); // tmp2 = mu*(M-Z)
        gsl_matrix_add(tmp2, Y); // tmp2 = Y + mu*(M-Z)
        gsl_matrix_memcpy(Y, tmp2); // Y = Y + mu*(M-Z)

        // calculation of some norms : the goal is PrimRes = norm(M-Z)/(norm(Z0)+eps) and DualRes = mu*norm(Z-Z0)/(norm(Z0)+eps)
        //Primres was partialy calculated few lines before
        tmp6 = matrix_norm(Z0) + epsilon;
        Primres /= tmp6; // PrimRes = norm(M-Z)/(norm(Z0)+eps)
        gsl_matrix_memcpy(tmp2, Z); // tmp2 = Z
        gsl_matrix_sub(tmp2, Z0); // tmp2 = Z-Z0
        Dualres = mu*matrix_norm(tmp2) / tmp6; // DualRes = mu*norm(Z-Z0)/(norm(Z0)+eps)

        // print the results only if verb == true
        if(verb)
        {
            printf("Iter = %d ; Primres = %1.15f ; DualRes = %1.15f ; mu = %1.15f \n",iter,Primres,Dualres,mu);
        }

        // verification of the convergence
        if(Primres < tol && Dualres < tol)
        {
            break;
        }
        else
        {
            if(Primres > 10 * Dualres)
            {
                mu = 2*mu;
            }
            else if(Dualres > 10 * Primres)
            {
                mu = mu/2;
            }
            else continue;
        }
    } // end of the loop

    gsl_matrix* R_test = gsl_matrix_calloc(3,3);
    gsl_matrix* R = gsl_matrix_calloc(2,3);
    gsl_matrix* c = gsl_matrix_calloc(1,1);
    syncRot(M,R_test,c);

    if(matrix_abs_sum(R_test) == 0) gsl_matrix_set_identity(R_test); // R = eyes

    // R = R(1:2,:);
    gsl_vector* R1 = gsl_vector_calloc(3);
    gsl_matrix_get_row(R1, R_test, 0);
    gsl_vector* R2 = gsl_vector_calloc(3);
    gsl_matrix_get_row(R2, R_test, 1);

    gsl_matrix_set_row(R,0,R1);
    gsl_matrix_set_row(R,1,R2);

    // S = Kroneckerproduct(C,eyes(3))*B , B(m*nb_joint);
    gsl_matrix* tmp7 = matrix_KPro(c,Im); // tmp7 = Kprod(C,Im)
    gsl_matrix* S = gsl_matrix_calloc(m,nb_joint);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tmp7, B, 0.0, S); // S = Kprod(C,Im)*B

    // part two of the iterations
    double fval = INFINITY;
    double C0;

    // In the matlab code they implemente a matrix C based on the size of pc, warning, C already exist, the new C is named Cnew
    gsl_matrix* Cnew;
    if (pc == nullptr)
    {
        Cnew = nullptr;
    }
    else
    {
        Cnew = gsl_matrix_calloc(1, (pc->size1)/3);
    }

    // initialization of the variables used during the iterations
    gsl_matrix *Rnew;
    double fvaltm1;
    gsl_matrix* tmp8 = gsl_matrix_alloc(R->size1, S->size2);
    gsl_matrix* tmp9 = gsl_matrix_alloc(tmp8->size1, D->size2);


    // iterations
    for (int iter = 0 ; iter < 1000 ; iter++) {
        // update translation : the goal is T = sum((W-R*S)*D,2)/(sum(diag(D))+eps) and W2fit = W - T*ones(1,p)
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, R, S, 0.0, tmp); // tmp = -R*S
        gsl_matrix_add(tmp, W_t); // tmp = W-R*S
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tmp, D, 0.0, tmp1); //tmp1 = (W-R*S)*D
        T = matrix_sum(tmp1, 2); // T = sum((W-R*S),2)
        gsl_vector_scale(T, 1 / (sum_D + epsilon)); // T = sum((W-R*S),2)/(sum(diad(D))+eps)

        // this for compute W2fit = W - T*ones(1,p);
        for (int i = 0; i < nb_joint; ++i) {
            for (int j = 0; j < T->size; j++) {
                gsl_matrix_set(W2fit, i, j, gsl_matrix_get(W, i, j) - gsl_vector_get(T, j));
            }
        }

        // update rotation : the goal is R = estimateR_weighted(S,W2fit,D,R);
        Rnew = estimateR_weighted(S, W2fit, D, R, verb);
        gsl_matrix_memcpy(R, Rnew); // update of R

        if (Cnew == nullptr) {
            C0 = estimateC_weighted(W2fit, R, B, D, 1e-3);
            gsl_matrix_memcpy(S, B);
            gsl_matrix_scale(S, C0);
        }


        fvaltm1 = fval; // fval = 0.5*norm((W2fit-R*S)*sqrt(D),'fro')^2 + 0.5*norm(C)^2;

        gsl_matrix_transpose_memcpy(tmp8, W2fit);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, R, S, 1.0, tmp8); // tmp8 = W2fit - R*S
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tmp8, matrix_cholesky(D), 0.0, tmp9); // tmp9 = (W2fit-R*S)*sqrt(D)
        fval = 0.5 * pow(matrix_norm(tmp9), 2) + 0.5 * pow(matrix_norm(Cnew), 2);

        // show output
        if (verb)
        {
            printf("Iter: %d, fval = %f\n", iter, fval);
        }

        // check convergence
        if (abs(fval-fvaltm1)/fvaltm1 < tol)
        {
            break;
        }
    } // end of the iterations

    // R(3,:) = cross(R(1,:),R(2,:));
    gsl_matrix* Rfinal = gsl_matrix_alloc(3,3);
    gsl_vector *row1 = gsl_vector_alloc(3);
    gsl_vector *row2 = gsl_vector_alloc(3);
    gsl_vector *row3 = gsl_vector_alloc(3);
    gsl_matrix_get_row(row1, R,0);
    gsl_matrix_get_row(row2, R,1);
    row3 = vector_cross_product(row1,row2);
    gsl_matrix_set_row(Rfinal,0,row1);
    gsl_matrix_set_row(Rfinal,1,row2);
    gsl_matrix_set_row(Rfinal,2,row3);


    pose output;
    output.S = S;
    output.M = M;
    output.R = Rfinal;
    output.C = Cnew;
    output.C0 = C0;
    output.T = T;
    output.fval = fval;

    return output;
}

// this function implements Pose FromKps_FP
pose pose_from_kps_FP(gsl_matrix* W, dictonnaire dict, gsl_matrix* R, gsl_vector* weight, bool verb, int lam_val, double tol)
{
    // this function solves
    // min ||W*diag(Z)-R*S-T||^2 + ||C||^2
    // where S = C1*B1+...+Cn*Bn,
    // Z denotes the depth of points

    //  weight.size, mu.size1,mu.size2,W.size1,W.size2,R.size1,R.size2,S.size1,S.size2

    const double epsilon = DBL_EPSILON;
    const short weightsize = weight->size;
    const short Wsize1 = W->size1;
    const short Wsize2 = W->size2;
    const short Rsize1 = R->size1;
    const short Rsize2 = R->size2;

    // centralize basis
    const gsl_matrix* mu = matrix_centralize(dict.mu);
    const gsl_matrix* pc = matrix_centralize(dict.pc);
    gsl_matrix* D = gsl_matrix_alloc(weightsize,weightsize);
    for (int i = 0; i < weightsize; ++i)
    {
        gsl_matrix_set(D,i,i,gsl_vector_get(weight,i));
    }

    const short musize1 = mu->size1;
    const short musize2 = mu->size2;

    // initialisation
    gsl_matrix* S = gsl_matrix_alloc(musize1,musize2);
    gsl_matrix_memcpy(S,mu);

    double C = 0;


    // T = mean(W,2)*mean(std(R(1:2,:)*S,1,2))/(mean(std(W,1,2))+eps);
    gsl_vector* T = matrix_sum(W,2);
    gsl_vector_scale(T,1.0/Wsize2);

    gsl_matrix* R_col12 = gsl_matrix_alloc(2,3);
    gsl_vector *row1 = gsl_vector_alloc(3);
    gsl_vector *row2 = gsl_vector_alloc(3);
    gsl_matrix_get_row(row1, R,0);
    gsl_matrix_get_row(row2, R,1);
    gsl_matrix_set_row(R_col12,0,row1);
    gsl_matrix_set_row(R_col12,1,row2);

    gsl_matrix* RS_col12 = matrix_product(R_col12,S);


    double meanRSstd = vector_sum(matrix_std(RS_col12,2))/2;
    double meanWstd = vector_sum(matrix_std(W,2))/Wsize1 + epsilon;

    gsl_vector_scale(T,meanRSstd/meanWstd); // mean(W,2)*mean(std(R(1:2,:)*S,1,2))/(mean(std(W,1,2))+eps)

    double fval = INFINITY;
    double fvaltm1;
    gsl_matrix* RS = gsl_matrix_alloc(Rsize1,musize2); // R*S
    gsl_matrix* RSplusT = gsl_matrix_alloc(Rsize1,musize2); // bsxfun(@plus,R*S,T)
    gsl_matrix* WRSplusT = gsl_matrix_alloc(Wsize1,musize2); // W.*bsxfun(@plus,R*S,T)
    gsl_vector* sumWRSplusT = gsl_vector_alloc(Wsize2); // sum(W.*bsxfun(@plus,R*S,T),1)
    gsl_matrix* Wsquare = matrix_square_elements(W); // W.^2
    gsl_vector* sumWsquare = matrix_sum(Wsquare,1); // (sum(W.^2,1)+eps)
    gsl_vector_add_constant(sumWsquare,epsilon);
    gsl_vector* Z = gsl_vector_alloc(Wsize2); // Z = sum(W.*bsxfun(@plus,R*S,T),1)./(sum(W.^2,1)+eps)
    gsl_matrix* Sp = gsl_matrix_alloc(Wsize1,Wsize2); // Sp = W*diag(Z)
    gsl_matrix* SpminusRS = gsl_matrix_alloc(Wsize1,Wsize2); // Sp-R*S
    gsl_matrix* SpminusRSD = gsl_matrix_alloc(Wsize1,weightsize); // (Sp-R*S)*D
    gsl_vector* sumSpminusRSD = gsl_vector_alloc(Wsize1); // sum((Sp-R*S)*D,2)
    double sumdiagD = vector_sum(matrix_diag(D))+epsilon; // sum(diag(D))+eps
    gsl_matrix* St = gsl_matrix_alloc(Wsize1,Wsize2); // bsxfun(@minus,Sp,T
    gsl_matrix* StD = gsl_matrix_alloc(Wsize1, weightsize); // St*D
    gsl_matrix* StDS = gsl_matrix_alloc(Wsize1, musize1); // St*D*S'
    gsl_matrix* U = gsl_matrix_alloc(Wsize1, musize1); // matrix_singular_value_decomposition(St*D*S', U, S, S_vect, V)
    gsl_matrix* _S = gsl_matrix_alloc(musize1, musize1);
    gsl_vector* _S_vect = gsl_vector_alloc(musize1);
    gsl_matrix* V = gsl_matrix_alloc(musize1, musize1);
    gsl_vector* diagchelou = gsl_vector_alloc(3);
    gsl_matrix* Udiagchelou = gsl_matrix_alloc(3,3);
    gsl_matrix* StminusRS = gsl_matrix_alloc(Wsize1,Wsize2); // (St-R*S)
    gsl_matrix* StminusRSsqrtD = gsl_matrix_alloc(Wsize1,weightsize); // (St-R*S)*sqrt(D)
    double signdetUV;
    // iterations
    for (int iter = 0 ; iter < 1000 ; ++iter)
    {
        // update of RS
        gsl_blas_dgemm(CblasNoTrans,CblasNoTrans, 1.0, R,S,0.0,RS);
        // update RS + T ( bsxfun(@plus,R*S,T) )
        for (int i = 0; i < Rsize1; ++i)
        {
            for (int j = 0; j < musize2; ++j)
            {
                gsl_matrix_set(RSplusT,i,j,gsl_matrix_get(RS,i,j) + gsl_vector_get(T,i));
            }
        }
        // update WRSplusT
        gsl_matrix_memcpy(WRSplusT,W);
        gsl_matrix_mul_elements(WRSplusT,RSplusT);
        // update sumWRSplusT
        sumWRSplusT = matrix_sum(WRSplusT,1);
        // update Z
        gsl_vector_memcpy(Z,sumWRSplusT);
        gsl_vector_div(Z,sumWsquare);

        // update R and T by aligning S to W*diag(Z);
        // update Sp
        gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,W,vector_diag(Z),0.0,Sp);
        //update SpminusRS
        gsl_matrix_memcpy(SpminusRS,Sp);
        gsl_matrix_sub(SpminusRS,RS);
        //update SpminusRSD
        gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,SpminusRS,D,0.0,SpminusRSD);
        // update sumSpminusRSD
        sumSpminusRSD = matrix_sum(SpminusRSD,2);
        // update of T
        gsl_vector_memcpy(T,sumSpminusRSD);
        gsl_vector_scale(T,1/sumdiagD);
        // update St
        for (int i = 0; i < Wsize1; ++i)
        {
            for (int j = 0; j < Wsize2; ++j)
            {
                gsl_matrix_set(St,i,j,gsl_matrix_get(Sp,i,j) - gsl_vector_get(T,i));
            }
        }
        //update StD
        gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,St,D,0.0,StD);
        //update StDS
        gsl_blas_dgemm(CblasNoTrans,CblasTrans,1.0,StD,S,0.0,StDS);
        //update U and V
        matrix_singular_value_decomposition(StDS, U, _S, _S_vect, V);
        //update R
        signdetUV = sign(matrix_det(matrix_product(U,matrix_transpose(V)))); // sign(det(U*V'))
        // update diagchelou
        gsl_vector_set_all(diagchelou,1);
        gsl_vector_set(diagchelou,2,signdetUV);
        // update Udiagchelou
        gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,U,vector_diag(diagchelou),0.0,Udiagchelou);
        // update R
        gsl_blas_dgemm(CblasNoTrans,CblasTrans,1.0,Udiagchelou,V,0.0,Udiagchelou);
        // update fvaltm1
        fvaltm1 = fval;
        // update StminusRS
        gsl_matrix_memcpy(StminusRS,St);
        gsl_matrix_sub(StminusRS,RS);
        // update StminusRSsqrtD
        gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,StminusRS,matrix_cholesky(D),0.0,StminusRSsqrtD);

        fval = pow(matrix_norm(StminusRSsqrtD),2);

        // show output
        if (verb)
        {
            printf("Iter: %d, fval = %f\n", iter, fval);
        }

        // check convergence
        if (abs(fval-fvaltm1)/fvaltm1 < tol)
        {
            break;
        }


    }

    pose output;
    output.S = S;
    output.R = R;
    output.C0 = 0;
    output.T = T;
    output.Z = Z;
    return output;

}

// this function takes an image as input and return the key points
pose find_maximal_response(int nb_joints)
{

    // name of the image
    char imname[] = "./../images_test/val_01_00_000000.bmp";

    // open the image
    Mat image;
    image = imread(imname, IMREAD_COLOR);


    // find the keypoints the keypoints first line is the values and the lignes 1 and 2 are the coordinates
    gsl_matrix* keypoints = gsl_matrix_calloc(nb_joints,3);
    gsl_vector* tmp;
    char keypointname[] = "./../images_test/val_01_00_000000_00.bmp";

    // computation of the maximums
    for(int i=0 ; i<nb_joints ; i++)
    {
        keypointname[35] = (char)(i+1)+48; // set the correct name
        tmp = find_max_image(imread(keypointname,0)); // open the image and find the max value and coordinates

        for(int j=0 ; j < 3 ; j++)
        {
            gsl_matrix_set(keypoints,i,j,gsl_vector_get(tmp,j));
        }
    }

    // set up the lens of the camera
    double lens_f = 319.4593;

    // then rescale the lens of the camera
    double lens_f_rescale = lens_f/640.0*64.0;

    // the camera has an offset we have to compensate
    double offset_i = -15.013/640.0*64.0;
    double offset_j = 64.8108/640.0*64.0;

    //computation of W_hp and W_hp_norm
    gsl_matrix* W_hp = gsl_matrix_calloc(nb_joints,2);
    gsl_matrix* W_hp_norm = gsl_matrix_calloc(nb_joints,3);

    for(int i=0 ; i<nb_joints ; i++)
    {
        gsl_matrix_set(W_hp,i,0, (gsl_matrix_get(keypoints,i,1) - offset_i));
        gsl_matrix_set(W_hp,i,1, (gsl_matrix_get(keypoints,i,2) - offset_j));

        gsl_matrix_set(W_hp_norm,i,0, (gsl_matrix_get(W_hp,i,0) - 32) / lens_f_rescale );
        gsl_matrix_set(W_hp_norm,i,1, (gsl_matrix_get(W_hp,i,1) - 32) / lens_f_rescale );
        gsl_matrix_set(W_hp_norm,i,2, 1);
    }

    // creation of the cad model and the dict
    model cad;
    dictonnaire dict;
    get_pascal_template(cad,dict);

    //initialization of the weight, verb, lam and tol that are arguments of pose_from_kps_WP
    gsl_vector* weight = gsl_vector_calloc(nb_joints);
    for (int j = 0 ; j < nb_joints ; j++)
    {
        gsl_vector_set(weight, j, gsl_matrix_get(keypoints,j,0));
    }

    pose output_wp = pose_from_kps_WP(W_hp, dict, weight, false, 1, 1e-10);
    pose output_fp = pose_from_kps_FP(matrix_transpose(W_hp_norm),dict,output_wp.R,weight,false,1,1e-10);

    return output_fp;


}


int main(void)
{
    clock_t start, stop;
    double totalTime;

    start = clock();

    pose output_fp;

    output_fp = find_maximal_response(8);

    printf("R\n");
    print_matrix(output_fp.R);
    printf("T\n");
    print_vector(output_fp.T);

    stop = clock();
    totalTime = (stop - start) / (double)CLOCKS_PER_SEC;



    printf("computation time = %f\n",totalTime);
    return 0;
}
