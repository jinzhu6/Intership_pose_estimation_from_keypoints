

#ifndef C_TRADUCTION_UTILS_STRUCT_H
#define C_TRADUCTION_UTILS_STRUCT_H

#endif //C_TRADUCTION_UTILS_STRUCT_H


#include <pwd.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <gsl/gsl_matrix.h>

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
    const static uint8_t nb_joints = 8; // number of joints of the model
    const static uint8_t m = 3; // dimension of the space
    gsl_matrix* B = gsl_matrix_alloc(m,nb_joints); // (I do not understand what it does, see the matlab code)
    gsl_matrix* mu = gsl_matrix_alloc(m,nb_joints); // (I do not understand what it does, see the matlab code)
    uint8_t kp_id[nb_joints] = {1,2,3,4,5,6,7,8}; // id of the joints (useless)
    std::string kp_name[nb_joints] = {"fru","frd","flu","fld","bru","brd","blu","bld"}; // names of the joints
    gsl_matrix* pc = nullptr; //(I think it is useless but it is used in pose_from_kps_WP, see the matlab code, here it is a empty array)

};


struct model
{
// this structure will be used to implement the cad model
public:
    const static int nb_faces = 1998; // number of faces of the cad model
    const static int nb_vertices = 1001; // number of vertices of the cad model
    const static uint8_t nb_joints = 8; // number of joints of the model
    const static uint8_t m = 3; // dimension of the space
    gsl_matrix* faces = gsl_matrix_alloc(nb_faces,m); // faces of the cad model
    gsl_matrix* vertices = gsl_matrix_alloc(nb_vertices,m); // vertices of the cad model
    std::string p_name[nb_joints] = {"fru","frd","flu","fld","bru","brd","blu","bld"}; // names of the joints of the model
    double scale[m] = {0.16,0.23,0.06}; // scales
    double kp[nb_joints][m] = {
            {scale[0]/2,scale[1]/2,scale[2]/2}    ,
            {scale[0]/2,scale[1]/2,-scale[2]/2}   ,
            {scale[0]/2,-scale[1]/2,scale[2]/2}   ,
            {scale[0]/2,-scale[1]/2,-scale[2]/2}  ,
            {-scale[0]/2,scale[1]/2,scale[2]/2}   ,
            {-scale[0]/2,scale[1]/2,-scale[2]/2}  ,
            {-scale[0]/2,-scale[1]/2,scale[2]/2}  ,
            {-scale[0]/2,-scale[1]/2,-scale[2]/2} ,
    }; // (I do not understand what it does, see the matlab code)
};

struct Store
{
// this struct implements the store structure in the matlab function estimateR_weight
public:
    gsl_matrix* E;
    bool isNullE = true;
    gsl_matrix* egrad;
    gsl_matrix* A;
    gsl_matrix* B;
    gsl_matrix* D;
};

struct Options
{
// this struct implements the options of the functions trustregion
public:
    //Problem problem; not used now
    int verbosity = 2;
    double maxtime = INFINITY;
    int miniter = 3;
    int maxiter = 1000;
    int mininner = 1;
    // int maxinner = problem.M.dim(); I hope it will not be used
    double tolgradnorm = 1e-6;
    double kappa = 0.1;
    double theta = 1.0;
    double rho_prime = 0.1;
    double rho_regularization = 1e3;
    double Deltabar = 1.4142;
    double Delta0 = 0.1768;
};

struct Info
{
// this struct store informations about the iteration in trustregion
public:
    int iter = 0;
    double cost = 6.2490;
    double gradnorm = 6.3054e-07;
    double Delta = 0.1768;
    double start_time;
    double time;
    double rho = INFINITY;
    double rhonum =  NAN;
    double rhoden = NAN;
    int accepted = 1;
    double numinner = NAN;
    double stepsize = NAN;
};
