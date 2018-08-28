

#ifndef C_TRADUCTION_UTILS_STRUCT_H
#define C_TRADUCTION_UTILS_STRUCT_H

#endif //C_TRADUCTION_UTILS_STRUCT_H


#include <pwd.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <gsl/gsl_matrix.h>

/*GSL*/
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>

using namespace cv;
using namespace std;


struct pose
{
// this structure will be used for the output of the function PoseFromKPTS_WP
public:
    gsl_matrix* S;
    gsl_matrix* M;
    gsl_matrix* R;
    gsl_matrix* C;
    double C0;
    gsl_vector* T;
    gsl_vector* Z;
    double fval;


};


struct dictonnaire
{
// this structure will be used to implement the function getPascalTemplate
public:
    uint8_t nb_joints; // number of joints of the model
    uint8_t m; // dimension of the space
    gsl_matrix* B;
    gsl_matrix* mu;
    gsl_matrix* pc;
};


struct model
{
// this structure will be used to implement the cad model
public:
    const static int nb_faces = 1998; // number of faces of the cad model
    const static int nb_vertices = 1001; // number of vertices of the cad model
    const static uint8_t nb_joints = 8; // number of joints of the model
    const static uint8_t dim = 3; // dimension of the space
    gsl_matrix* faces = gsl_matrix_alloc(nb_faces,dim); // faces of the cad model
    gsl_matrix* vertices = gsl_matrix_alloc(nb_vertices,dim); // vertices of the cad model
    std::string p_name[nb_joints] = {"fru","frd","flu","fld","bru","brd","blu","bld"}; // names of the joints of the model
    double scale[dim] = {0.16,0.23,0.06}; // scales
    double kp[nb_joints][dim] = {
            {scale[0]/2,scale[1]/2,scale[2]/2}    ,
            {scale[0]/2,scale[1]/2,-scale[2]/2}   ,
            {scale[0]/2,-scale[1]/2,scale[2]/2}   ,
            {scale[0]/2,-scale[1]/2,-scale[2]/2}  ,
            {-scale[0]/2,scale[1]/2,scale[2]/2}   ,
            {-scale[0]/2,scale[1]/2,-scale[2]/2}  ,
            {-scale[0]/2,-scale[1]/2,scale[2]/2}  ,
            {-scale[0]/2,-scale[1]/2,-scale[2]/2} ,
    }; // position of the kp in the first referenciel
};
