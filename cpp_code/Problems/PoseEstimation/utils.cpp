#include "Problems/PoseEstimation/utils.h"

using namespace cv;
using namespace std;

// this function prints a gsl matrix
int print_matrix(const gsl_matrix *mat)
{
    if (mat == nullptr)
    {
        printf("matrix null\n");
        return 0;
    }

    int m = mat->size1;
    int n = mat->size2;

    for (int i = 0; i < m; i++)
    {
        printf("[ ");
        for (int j = 0; j < n; j++)
        {
            printf(" %.4f  ", gsl_matrix_get(mat,i,j));
        }
        printf(" ]\n");
    }
    printf("\n");
    return 0;
}

// this function prints a gsl vector
int print_vector(const gsl_vector* vect)
{
    if (vect == nullptr)
    {
        printf("vector null\n");
        return 0;
    }
    int m = vect->size;

    for (int i = 0; i < m; i++)
    {
        printf("[ %.4f ] ", gsl_vector_get(vect,i));
        printf("\n");
    }
    printf("\n");
    return 0;
}

// this function computes the sum of the following the specified axis
gsl_vector* matrix_sum(gsl_matrix* mat, uint8_t axis)
{
    uint8_t size1 = mat->size1;
    uint8_t size2 = mat->size2;
    if(axis==2)
    {
        gsl_vector* res = gsl_vector_calloc(size1);
        for(int i = 0 ; i < size1 ; i++)
        {
            for(int j = 0 ; j <size2 ; j++)
            {
                gsl_vector_set(res,i,gsl_vector_get(res,i)+gsl_matrix_get(mat,i,j));
            }
        }
        return res;
    }
    else if(axis==1)
    {
        gsl_vector* res = gsl_vector_calloc(size2);
        for(int i = 0 ; i < size2 ; i++)
        {
            for(int j = 0 ; j <size1 ; j++)
            {
                gsl_vector_set(res,i,gsl_vector_get(res,i)+gsl_matrix_get(mat,j,i));
            }
        }
        return res;
    }
    else
    {
        printf("axis should be 1 or 2. Axis given : %d",axis);
        return gsl_vector_calloc(0);
    }
}

// this function returns the sum of all the element of the vector vect
double vector_sum(gsl_vector* vect)
{
    double res = 0;
    for (int i = 0; i < vect->size ; ++i)
    {
        res += gsl_vector_get(vect,i);
    }
    return res;
}

// this function returns the inverse of mat, mat should be square and non singular
// for the algorithm Pose from Kpts, the singularity is not a issue because the probability of find a singular matrix between all the reel matrix is egal to zero
gsl_matrix* matrix_inversion(gsl_matrix* mat)
{
    if (mat == nullptr)
    {
        return nullptr;
    }

    const uint8_t n = mat->size1;
    const uint8_t m = mat->size2;
    gsl_permutation* p = gsl_permutation_calloc(n);
    int s;
    gsl_matrix* inv = gsl_matrix_calloc(n,m);
    if (n != m)
    {
        printf("the matrix given in argument is not square");
        return inv;
    }
    //compute the LU decomposition of mat
    gsl_linalg_LU_decomp(mat, p, &s);

    //compute the inverse of the LU decomposition
    gsl_linalg_LU_invert(mat, p, inv);
    gsl_permutation_free(p);

    return inv;
}

// this function computes the Frobenius norm of the matrix
double matrix_norm(gsl_matrix* A)
{
    if (A == nullptr)
    {
        return 0;
    }

    double res = 0;
    uint8_t m = A->size1;
    uint8_t n = A->size2;
    for (int i = 0 ; i < m ; i++)
    {
        for (int j = 0; j < n ; j++)
        {
            res += pow(gsl_matrix_get(A,i,j),2);
        }
    }
    return sqrt(res);
}

// this function returns the trace of A, A has to be square
double matrix_trace(gsl_matrix* A)
{
    if (A == nullptr)
    {
        return 0;
    }

    uint8_t m = A->size1;
    uint8_t n = A->size2;
    if(n != m)
    {
        printf("the matrix must be square");
        return raise(SIGILL);
    }
    double trace = 0;
    for (int i = 0 ; i < n ; i++)
    {
        trace += gsl_matrix_get(A,i,i);
    }
    return trace;
}

// this function normalizes a array 3*8
int normalizeS(gsl_matrix* res)
{
    double tab_mean[3] = {0,0,0};
    double tab_std[3] = {0,0,0};

    // computation of the mean
    for(int i = 0 ; i < 3 ; i++)
    {
        for(int j = 0 ; j < 8 ; j++)
        {
            tab_mean[i] += gsl_matrix_get(res,i,j) / 8;
        }
    }

    // modification of S
    for(int i = 0 ; i < 3 ; i++)
    {
        for(int j = 0 ; j < 8 ; j++)
        {
            gsl_matrix_set(res,i,j,gsl_matrix_get(res,i,j) - tab_mean[i]);
        }
    }

    // computation of the standard deviation
    for(int i = 0 ; i < 3 ; i++)
    {
        for(int j = 0 ; j < 8 ; j++)
        {
            tab_std[i] += pow(gsl_matrix_get(res,i,j),2) / 8;
        }
        tab_std[i] = sqrt( tab_std[i] - pow(tab_mean[i],2) );
    }

    // modification of S
    double std = (tab_std[0] + tab_std[1] + tab_std[2])/3;
    for(int i = 0 ; i < 3 ; i++)
    {
        for(int j = 0 ; j < 8 ; j++)
        {
            gsl_matrix_set(res,i,j,gsl_matrix_get(res,i,j) / std);
        }
    }

    return 0;
}

// this function takes an image as input and return the value and coordinate of the maximum
// res[0] : x , res[1] : y , res[2] : values
gsl_vector* find_max_image(Mat image)
{
    uint8_t *myData = image.data;
    float val = 0;
    int width = image.size().width;
    int height = image.size().height;

    int _stride = image.step;
    gsl_vector* res = gsl_vector_calloc(3);

    for(uint8_t i = 0; i < height; i++)
    {
        for(uint8_t j = 0; j < width; j++)
        {
            //val = myData[ i * _stride + j];
            Scalar intensity = image.at<uchar>(j, i);
            val = intensity.val[0]/255.0;

            if(val > gsl_vector_get(res,0))
            {
                gsl_vector_set(res, 0, val);
                gsl_vector_set(res, 1, i+1);
                gsl_vector_set(res, 2, j+1); //TODo : supprimer les +1 qund le debug est fini
            }
        }
        //TODO : pourquoi il lit pqs bien les images bordel
    }
    return res;
}

// this function reads a cvs file and modifies the array data, it works, do not touch it !
//the file sould has 3 rows
int read_csv(char *filename, double data[][3])
{
    FILE* my_file = fopen(filename,"r");


    if (!my_file) { /* open operation failed. */
        perror("Failed opening file '/home/me/data.txt' for reading:");
        exit(1);
    }

    char buffer[100];
    char* buffer2;

    int i = 0, j = 0;

    // this for read the csv file
    while (fscanf(my_file, "%s", buffer)!=EOF)
    {
        buffer2 = strtok (buffer,",");
        while (buffer2 != NULL)
        {
            data[i][j] = atof(buffer2);
            //printf("%s ;", buffer2);
            buffer2 = strtok (NULL, ",");
            j++;
        }
        //printf("\n");
        j = 0;
        i++;
    }
    fclose(my_file);

    return 0;


}


// this function computes the singular value decomposition of A(m*n), the output are U,V, S and S_vect
// A = USV' ; U : m*n (orthogonal) ; S : n*n (diagonal) ; V : n*n (orthogonal square) ; S_vect is just the vector version of S
int matrix_singular_value_decomposition(gsl_matrix* A, gsl_matrix* U, gsl_matrix* S, gsl_vector* S_vect,  gsl_matrix* V)
{
    // size of the matrix A
    uint8_t m = A->size1;
    uint8_t n = A->size2;
    gsl_matrix_memcpy(U,A); // U = A
    gsl_vector_set_zero(S_vect); // setting S_vect at zero
    gsl_matrix_set_zero(S); // setting V and S at zero
    gsl_matrix_set_zero(V);
    gsl_vector* v = gsl_vector_calloc(n); // this vector is compulsory for gsl_linalg_SV_decomp
    gsl_linalg_SV_decomp(U,V,S_vect,v); // computation of the singular value decomposition

    // this loop sets the value of S
    for (int i = 0 ; i < n ; i++)
    {
        gsl_matrix_set(S, i, i, gsl_vector_get(S_vect,i));
    }
    gsl_vector_free(v); // free v
    return 0;
}

// this function returns the max of abs(A)
// if A = [-8,5,7,-4] the function will return -8
double matrix_max_abs(gsl_matrix* A)
{
    double res = 0;
    double tmp = 0;
    uint8_t m = A->size1;
    uint8_t n = A->size2;
    for (int i = 0 ; i < m; i++)
    {
        for (int j = 0 ; j < n ; j++)
        {
            tmp = gsl_matrix_get(A,i,j);
            if(abs(res) < abs(tmp)) res = tmp;
        }
    }
    return res;
}

// this function returns the sum of all the absolute value of the elements of A
double matrix_abs_sum(gsl_matrix* A)
{
    double sum = 0;
    uint8_t m = A->size1;
    uint8_t n = A->size2;
    for (int i = 0 ; i < m; i++)
    {
        for (int j = 0 ; j < n ; j++)
        {
            sum += abs(gsl_matrix_get(A,i,j));
        }
    }
    return sum;
}

// this function returns the kronecker product between a(m*p) and b(n*q), the returned matrix will be m*n by p*q
gsl_matrix* matrix_KPro(gsl_matrix* a, gsl_matrix* b) {
    int i, j, k, l;
    int m, p, n, q;
    m = a->size1;
    p = a->size2;
    n = b->size1;
    q = b->size2;

    gsl_matrix* c = gsl_matrix_alloc(m*n, p*q);
    double da, db;

    for (i = 0 ; i < m ; i++)    {
        for (j = 0 ; j < p ; j++)   {
            da = gsl_matrix_get (a, i, j);
            for (k = 0 ; k < n ; k++)   {
                for (l = 0 ; l < q ; l++)   {
                    db = gsl_matrix_get (b, k, l);
                    gsl_matrix_set (c, n*i+k, q*j+l, da * db);
                }
            }
        }
    }
    return c;
}

// this function returns the cross product of u by v, u and v must have a lenght of 3
gsl_vector* vector_cross_product(const gsl_vector* u, const gsl_vector* v)
{
    gsl_vector* product = gsl_vector_calloc(3);

    double p1 = gsl_vector_get(u, 1)*gsl_vector_get(v, 2)
                - gsl_vector_get(u, 2)*gsl_vector_get(v, 1);

    double p2 = gsl_vector_get(u, 2)*gsl_vector_get(v, 0)
                - gsl_vector_get(u, 0)*gsl_vector_get(v, 2);

    double p3 = gsl_vector_get(u, 0)*gsl_vector_get(v, 1)
                - gsl_vector_get(u, 1)*gsl_vector_get(v, 0);

    gsl_vector_set(product, 0, p1);
    gsl_vector_set(product, 1, p2);
    gsl_vector_set(product, 2, p3);

    return product;
}

// this functions returns the dot product of u by v
double vector_dot_product(const gsl_vector* u, const gsl_vector* v)
{
    uint8_t m = u->size;
    uint8_t n = v->size;
    if(n != m)
    {
        printf("the vectors must have the same lenght");
        raise(SIGILL);
    }
    double dot_product = 0;
    for (int i = 0 ; i < n ; i++)
    {
        dot_product += gsl_vector_get(u,i) * gsl_vector_get(v,i);
    }
    return dot_product;
}

// this function returns the inner procuct between A and B, A and B must have the same size
double matrix_inner_product(gsl_matrix* A, gsl_matrix* B)
{
    double sum = 0;
    uint8_t mA = A->size1;
    uint8_t nA = A->size2;
    uint8_t mB = B->size1;
    uint8_t nB = B->size2;
    if(nA != nB or mA != mB)
    {
        printf("the matrix must have the same size");
        raise(SIGILL);
    }
    for (int i = 0 ; i < mA; i++)
    {
        for (int j = 0 ; j < nA ; j++)
        {
            sum += gsl_matrix_get(A,i,j) * gsl_matrix_get(B,i,j);
        }
    }
    return sum;
}

// this function pseudo-inverse of A using the SVD
gsl_matrix* matrix_pinv(gsl_matrix* A)
{
    int Asize1 = A->size1, Asize2 = A->size2;
    double x;
    gsl_matrix *result = gsl_matrix_alloc(Asize2, Asize1);

    if (Asize1 == 1 and Asize1 == 1)
    {
        x = gsl_matrix_get(A,0,0);
        if (x != 0)
        {
            gsl_matrix_set(result,0,0,1/x);
        }
    }
    else
    {

        gsl_matrix *tmp = gsl_matrix_alloc(Asize2,Asize2);
        gsl_matrix *U = gsl_matrix_alloc(Asize1, Asize2);
        gsl_matrix *V = gsl_matrix_alloc(Asize2, Asize2);
        gsl_vector *S_vect = gsl_vector_alloc(Asize2);
        gsl_matrix* S = gsl_matrix_alloc(Asize2,Asize2);
        matrix_singular_value_decomposition(A, U, S, S_vect, V);
        for(int i = 0 ; i < A->size2 ; i++)
        {
            x = gsl_vector_get(S_vect,i);
            if (x != 0)
            {
                gsl_matrix_set(S,i,i,1/x);
            }
        }

        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, V, S, 0.0, tmp); // tmp = V*S^+
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, tmp, U, 0.0, result); // tmp = V*S^+
    }
    return result;
}

// this function returns L : the Cholesky decomposition of the matrix A. A = LL'
gsl_matrix* matrix_cholesky(gsl_matrix* A)
{
    gsl_matrix* res = gsl_matrix_alloc(A->size1,A->size2);
    gsl_matrix_memcpy(res,A);
    gsl_linalg_cholesky_decomp(res);
    return res;
}

// returns A*B
gsl_matrix* matrix_product(const gsl_matrix* A, const gsl_matrix* B)
{
    gsl_matrix* res = gsl_matrix_alloc(A->size1,B->size2);
    gsl_blas_dgemm(CblasNoTrans,CblasNoTrans, 1.0,A,B,0.0,res);
    return res;
}

// returns A'
gsl_matrix* matrix_transpose(const gsl_matrix* A)
{
    gsl_matrix* res = gsl_matrix_alloc(A->size2,A->size1);
    gsl_matrix_transpose_memcpy(res,A);
    return res;
}

// returns centralised A
gsl_matrix* matrix_centralize(gsl_matrix* A)
{
    if(A == nullptr)
        return nullptr;
    else
    {
        gsl_vector *meanA = matrix_sum(A, 2);
        gsl_vector_scale(meanA, 1 / A->size2);
        gsl_matrix *res = gsl_matrix_alloc(A->size1, A->size2);
        double x;
        for (int i = 0; i < A->size1; ++i) {
            for (int j = 0; j < A->size2; ++j) {
                x = gsl_matrix_get(A, i, j) - gsl_vector_get(meanA, i);
                gsl_matrix_set(res, i, j, x);
            }
        }
        return res;
    }
}

// returns the std of A following the axis axis
gsl_vector* matrix_std(const gsl_matrix* mat, int axis)
{
    uint8_t size1 = mat->size1;
    uint8_t size2 = mat->size2;
    // axis 2
    if(axis==2)
    {
        gsl_vector* meanA = gsl_vector_calloc(size1);
        gsl_vector* meansquareA = gsl_vector_calloc(size1);
        for(int i = 0 ; i < size1 ; i++)
        {
            for(int j = 0 ; j <size2 ; j++)
            {
                gsl_vector_set(meanA,i,gsl_vector_get(meanA,i)+gsl_matrix_get(mat,i,j));
                gsl_vector_set(meansquareA,i,gsl_vector_get(meansquareA,i)+pow(gsl_matrix_get(mat,i,j),2));
            }
            gsl_vector_set(meanA,i,pow(gsl_vector_get(meanA,i),2));
        }
        gsl_vector_scale(meanA,1.0/pow(size2,2));
        gsl_vector_scale(meansquareA,1.0/size2);

        gsl_vector_sub(meansquareA,meanA);
        for (int i = 0 ; i < size1 ; ++i)
        {
            gsl_vector_set(meansquareA,i,sqrt(gsl_vector_get(meansquareA,i)));
        }
        return meansquareA;

    }

    else if(axis==1)
    {
        gsl_vector* meanA = gsl_vector_calloc(size2);
        gsl_vector* meansquareA = gsl_vector_calloc(size2);
        for(int i = 0 ; i < size2 ; i++)
        {
            for(int j = 0 ; j <size1 ; j++)
            {
                gsl_vector_set(meanA,i,gsl_vector_get(meanA,i)+gsl_matrix_get(mat,j,i));
                gsl_vector_set(meansquareA,i,gsl_vector_get(meansquareA,i)+pow(gsl_matrix_get(mat,j,i),2));
            }
            gsl_vector_set(meanA,i,pow(gsl_vector_get(meanA,i),2));
        }
        gsl_vector_scale(meanA,1.0/pow(size1,2));
        gsl_vector_scale(meansquareA,1.0/size1);

        gsl_vector_sub(meansquareA,meanA);
        for (int i = 0 ; i < size2 ; ++i)
        {
            gsl_vector_set(meansquareA,i,sqrt(gsl_vector_get(meansquareA,i)));
        }
        return meansquareA;
    }
    else
    {
        printf("axis should be 1 or 2. Axis given : %d",axis);
        return nullptr;
    }
}

// return the term - to - term squared elements
gsl_matrix* matrix_square_elements(const gsl_matrix* A)
{
    if (A == nullptr)
        return nullptr;
    else
    {
        gsl_matrix *res = gsl_matrix_alloc(A->size1, A->size2);
        for (int i = 0; i < A->size1; ++i) {
            for (int j = 0; j < A->size2; ++j) {
                gsl_matrix_set(res, i, j, pow(gsl_matrix_get(A, i, j), 2));
            }
        }
        return res;
    }
}

// returns the diagonal matrix of v
gsl_matrix* vector_diag(const gsl_vector* v)
{
    if(v == nullptr) return nullptr;
    else
    {
        gsl_matrix* res = gsl_matrix_calloc(v->size,v->size);
        for (int i = 0; i < v->size; ++i)
        {
            gsl_matrix_set(res,i,i,gsl_vector_get(v,i));
        }
        return res;
    }
}

// returns the vector of the diagonal of m
gsl_vector* matrix_diag(const gsl_matrix* m)
{
    if(m == nullptr) return nullptr;
    else
    {
        gsl_vector* res = gsl_vector_calloc(m->size1);
        for (int i = 0; i < m->size1; ++i)
        {
            gsl_vector_set(res,i,gsl_matrix_get(m,i,i));
        }
        return res;
    }
}

// sign function
double sign(double a)
{
    if(a == 0) return 0.0;
    else if(a > 0) return 1.0;
    else return -1.0;
}

// returns det(A)
double matrix_det(gsl_matrix *A)
{
    double det;
    int signum;
    gsl_permutation *p = gsl_permutation_alloc(A->size1);

    gsl_matrix *tmpA = gsl_matrix_alloc(A->size1, A->size2);
    gsl_matrix_memcpy(tmpA , A);


    gsl_linalg_LU_decomp(tmpA , p , &signum);
    det = gsl_linalg_LU_det(tmpA , signum);
    gsl_permutation_free(p);
    gsl_matrix_free(tmpA);


    return det;
}
