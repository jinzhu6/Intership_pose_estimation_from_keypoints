# Makefile for ROPTLIB. Test on ubuntu 16.04 LTS

# set compiler
CC = g++
CXXFLAGS:=-O3 -ffastmath -march=native -ggdb3

# default test problem is the Brockett cost function on the Stiefel manifold
TP?=TestPoseEstimation

#the path of ROPTLIB
ROOTPATH:=$(dir $(realpath $(firstword $(MAKEFILE_LIST))))

# directories of ROPTLIB header files
INCDIRS = -I$(ROOTPATH)/
INCDIRS += -I$(ROOTPATH)/Manifolds/
INCDIRS += -I$(ROOTPATH)/Manifolds/Stiefel/
INCDIRS += -I$(ROOTPATH)/Others/
INCDIRS += -I$(ROOTPATH)/Problems/
INCDIRS += -I$(ROOTPATH)/Problems/PoseEstimation/
INCDIRS += -I$(ROOTPATH)/Solvers/
INCDIRS += -I$(ROOTPATH)/cwrapper/
INCDIRS += -I$(ROOTPATH)/cwrapper/blas/
INCDIRS += -I$(ROOTPATH)/cwrapper/lapack/

# ROPTLIB C++ files
CPPS += $(ROOTPATH)/Manifolds/Element.cpp $(ROOTPATH)/Manifolds/LinearOPE.cpp $(ROOTPATH)/Manifolds/Manifold.cpp $(ROOTPATH)/Manifolds/ProductElement.cpp $(ROOTPATH)/Manifolds/ProductManifold.cpp $(ROOTPATH)/Manifolds/SharedSpace.cpp $(ROOTPATH)/Manifolds/SmartSpace.cpp 
CPPS += $(ROOTPATH)/Manifolds/Euclidean/EucVariable.cpp $(ROOTPATH)/Manifolds/Euclidean/EucVector.cpp $(ROOTPATH)/Manifolds/Euclidean/Euclidean.cpp 
CPPS += $(ROOTPATH)/Manifolds/Grassmann/GrassVariable.cpp $(ROOTPATH)/Manifolds/Grassmann/GrassVector.cpp $(ROOTPATH)/Manifolds/Grassmann/Grassmann.cpp 
CPPS += $(ROOTPATH)/Manifolds/L2Sphere/L2Sphere.cpp $(ROOTPATH)/Manifolds/L2Sphere/L2SphereVariable.cpp $(ROOTPATH)/Manifolds/L2Sphere/L2SphereVector.cpp 
CPPS += $(ROOTPATH)/Manifolds/OrthGroup/OrthGroup.cpp $(ROOTPATH)/Manifolds/OrthGroup/OrthGroupVariable.cpp $(ROOTPATH)/Manifolds/OrthGroup/OrthGroupVector.cpp 
CPPS += $(ROOTPATH)/Manifolds/SPDManifold/SPDManifold.cpp $(ROOTPATH)/Manifolds/SPDManifold/SPDVariable.cpp $(ROOTPATH)/Manifolds/SPDManifold/SPDVector.cpp 
CPPS += $(ROOTPATH)/Manifolds/SPDTensor/SPDTVariable.cpp $(ROOTPATH)/Manifolds/SPDTensor/SPDTVector.cpp $(ROOTPATH)/Manifolds/SPDTensor/SPDTensor.cpp 
CPPS += $(ROOTPATH)/Manifolds/Sphere/Sphere.cpp $(ROOTPATH)/Manifolds/Sphere/SphereVariable.cpp $(ROOTPATH)/Manifolds/Sphere/SphereVector.cpp 
CPPS += $(ROOTPATH)/Manifolds/Stiefel/StieVariable.cpp $(ROOTPATH)/Manifolds/Stiefel/StieVector.cpp $(ROOTPATH)/Manifolds/Stiefel/Stiefel.cpp 
CPPS += $(ROOTPATH)/Others/ForDebug.cpp $(ROOTPATH)/Others/MinPNormConHull.cpp $(ROOTPATH)/Others/MyMatrix.cpp $(ROOTPATH)/Others/Spline.cpp $(ROOTPATH)/Others/Timer.cpp $(ROOTPATH)/Others/randgen.cpp 
CPPS += $(ROOTPATH)/Problems/Problem.cpp
CPPS += $(ROOTPATH)/Problems/ElasticCurvesRO/DriverElasticCurvesRO.cpp $(ROOTPATH)/Problems/ElasticCurvesRO/ElasticCurvesRO.cpp
CPPS += $(ROOTPATH)/Problems/PreShapePathStraighten/PreShapePathStraighten.cpp 
CPPS += $(ROOTPATH)/Problems/SPDMean/SPDMean.cpp 
CPPS += $(ROOTPATH)/Problems/SPDTensorDL/SPDTensorDL.cpp 
CPPS += $(ROOTPATH)/Problems/SphereConvexHull/SphereConvexHull.cpp
CPPS += $(ROOTPATH)/Problems/PoseEstimation/PoseEstimation.cpp
CPPS += $(ROOTPATH)/Problems/PoseEstimation/utils.cpp
CPPS += $(ROOTPATH)/Solvers/LRBFGS.cpp $(ROOTPATH)/Solvers/LRTRSR1.cpp $(ROOTPATH)/Solvers/QuasiNewton.cpp $(ROOTPATH)/Solvers/RBFGS.cpp $(ROOTPATH)/Solvers/RBroydenFamily.cpp $(ROOTPATH)/Solvers/RCG.cpp $(ROOTPATH)/Solvers/RGS.cpp $(ROOTPATH)/Solvers/RNewton.cpp $(ROOTPATH)/Solvers/RSD.cpp $(ROOTPATH)/Solvers/RTRNewton.cpp $(ROOTPATH)/Solvers/RTRSD.cpp $(ROOTPATH)/Solvers/RTRSR1.cpp $(ROOTPATH)/Solvers/RWRBFGS.cpp $(ROOTPATH)/Solvers/Solvers.cpp $(ROOTPATH)/Solvers/SolversLS.cpp $(ROOTPATH)/Solvers/SolversLSLPSub.cpp $(ROOTPATH)/Solvers/SolversTR.cpp 

# convert a string to upper case.
UPPER_TP  = $(shell echo $(TP) | tr a-z A-Z)

# make a binary file, which is called in command line
ROPTLIB:
	$(CC) -O3 -w -std=c++0x $(ROOTPATH)test/$(TP).cpp $(CPPS) $(INCDIRS) -D$(UPPER_TP) -llapack -lblas -lgsl `pkg-config --cflags --libs opencv` -lm -o $(TP)

