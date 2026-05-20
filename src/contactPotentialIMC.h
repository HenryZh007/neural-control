#ifndef CONTACTPOTENTIALIMC_H
#define CONTACTPOTENTIALIMC_H

#include "eigenIncludes.h"
#include "elasticRod.h"
#include "timeStepper.h"
#include "collisionDetector.h"

#include <symengine/llvm_double.h>

using namespace SymEngine;


class contactPotentialIMC
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    contactPotentialIMC(elasticRod &m_rod, timeStepper &m_stepper, double m_d_h, 
        double m_col_limit, double m_k_scaler = 5e2);

    ~contactPotentialIMC() = default;
    void updateContactStiffness();
    void computeFcJc();

    void updateContactInfo(const MatrixXd &contact_info);

    void constructCandidateSet();

    void computeFcJC();

private:
    elasticRod* rod;
    timeStepper* stepper;

    vector<Vector3d> contact_meshes;

    vector<Vector2i> candidate_set;
    

    double d_h;
    double col_limit;
    double k_scaler;
    double scale;

    Vector<double, 8> contact_input;
    Vector<double, 3> contact_gradient;
    Matrix<double, 3, 3> contact_hessian;


    RCP<const Basic> x_x;
    RCP<const Basic> x_y;
    RCP<const Basic> x_z;

    RCP<const Basic> c_x;
    RCP<const Basic> c_y;
    RCP<const Basic> c_z;

    RCP <const Basic> d_h_Bar;
    RCP <const Basic> R;

    void generateEqs();
    void get_norm(const DenseMatrix &num, RCP<const Basic> &C);
    void subtract_matrix(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C);

    LLVMDoubleVisitor E_gradient_func;
    LLVMDoubleVisitor E_hessian_func;
    LLVMDoubleVisitor E_dist_func;

};

#endif