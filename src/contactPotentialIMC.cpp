#include "contactPotentialIMC.h"

contactPotentialIMC::contactPotentialIMC(elasticRod &m_rod, timeStepper &m_stepper, 
                                             double m_d_h, double m_col_limit, double m_k_scaler){
    rod = &m_rod;
    stepper = &m_stepper;
    d_h = 1.0;
    col_limit = m_col_limit;
    k_scaler = m_k_scaler;
    scale = 1.0 / m_d_h;
    col_limit *= scale; // scale the col_limit to match the d_h scaling

    contact_input.setZero();      // MUST ensure this is valid Eigen::VectorXd of size 8
    contact_gradient.setZero();   // same
    contact_hessian.setZero();    // same

    generateEqs();
    
    // Vector<double, 8> wall_friction_input;
    contact_input[6] = 1.0;
}


void contactPotentialIMC::updateContactInfo(const MatrixXd &contact_info)
{
    // This function should update the contact information based on the current state of the rod
    // and the environment.
    // For now, we will just print a message.
    contact_meshes.clear();

    for (int i = 0; i < contact_info.rows(); i++)
    {
        Vector3d contact_point(contact_info(i, 0), contact_info(i, 1), contact_info(i, 2));
        contact_meshes.push_back(contact_point);
    }
}

void contactPotentialIMC::constructCandidateSet()
{
    candidate_set.clear();
    double min_dist = 1e4;
    for (int i = 0; i < rod->nv; i++)
    {
        Vector3d xLocal = rod->getVertex(i);
        for (int j = 0; j < contact_meshes.size(); j++)
        {
           Vector3d center(contact_meshes[j](0), 0, contact_meshes[j](1));
           double r = contact_meshes[j](2) * scale;
           contact_input[7] = r;
           contact_input(seq(0, 2)) = xLocal * scale;
           contact_input(seq(3, 5)) = center * scale;
           double dist;
           E_dist_func.call(&dist, contact_input.data());

           if (dist < min_dist)
           {
                min_dist = dist;
           }
           
           if (dist < col_limit)
           {
                candidate_set.push_back(Vector2i(i, j));
           }
        }
    }
    // cout <<"min_dist: " << min_dist <<" " << col_limit<< endl;

    // if (candidate_set.size() > 0)
    // {
    //     cout << "Candidate set size: " << candidate_set.size() << endl;
    //     exit(0);
    // }

}

void contactPotentialIMC::computeFcJc()
{
    int num_collisions = 0;
    for (const auto &pair : candidate_set)
    {
        int rod_index = pair(0);
        int mesh_index = pair(1);
        Vector3d xLocal = rod->getVertex(rod_index);
        Vector3d center(contact_meshes[mesh_index](0), 0, contact_meshes[mesh_index](1));
        double r = contact_meshes[mesh_index](2) * scale;
        contact_input[7] = r;
        contact_input(seq(0, 2)) = xLocal * scale;
        contact_input(seq(3, 5)) = center * scale;
        
        double dist;
        E_dist_func.call(&dist, contact_input.data());

        // cout <<"dist: " << dist << endl;

        contact_gradient.setZero();
        contact_hessian.setZero();

        if (dist < 1.0)
        {
            E_gradient_func.call(contact_gradient.data(), contact_input.data());
            E_hessian_func.call(contact_hessian.data(), contact_input.data());
            num_collisions++;
        }

        contact_gradient *= k_scaler;
        contact_hessian *= scale * k_scaler;

        // cout << "contact_gradient: "  << contact_gradient.norm() << endl;

        for (int i = 0; i < 3; i++)
        {
            stepper->addForce(rod_index * 4 + i, contact_gradient(i));
            for (int j = 0; j < 3; j++)
            {
                stepper->addJacobian(rod_index * 4 + i, rod_index * 4 + j, contact_hessian(i, j));
            }
        }
    }
    // cout << "Number of collisions: " << num_collisions << endl;



}







void contactPotentialIMC::generateEqs()
{
    x_x = symbol("x_x");
    x_y = symbol("x_y");
    x_z = symbol("x_z");
    c_x = symbol("c_x");
    c_y = symbol("c_y");
    c_z = symbol("c_z");
    d_h_Bar = symbol("d_h");
    R = symbol("R");

    bool symbolic_cse = true;
    int opt_level = 3;

    DenseMatrix x({x_x, x_y, x_z});
    DenseMatrix c({c_x, c_y, c_z});

    DenseMatrix tmp(3, 1);
    subtract_matrix(x, c, tmp);

    RCP <const Basic> dist;
    get_norm(tmp, dist);
    RCP <const Basic> d = sub(dist,  R);

    RCP <const Basic> E;

    E = mul(pow(sub(d, d_h_Bar), 2), log(div(d_h_Bar, d)));

    DenseMatrix E_potential({E});
    DenseMatrix Dist({d});

    DenseMatrix E_gradient(1, 3);
    DenseMatrix E_hessian(3, 3);

    vec_basic nodes_vec {x_x, x_y, x_z};
    DenseMatrix nodes {nodes_vec};

    vec_basic func_inputs(nodes_vec);
    func_inputs.push_back(c_x);
    func_inputs.push_back(c_y);
    func_inputs.push_back(c_z);
    func_inputs.push_back(d_h_Bar);
    func_inputs.push_back(R);

    jacobian(E_potential, nodes, E_gradient);
    jacobian(E_gradient, nodes, E_hessian);

    E_gradient_func.init(func_inputs, E_gradient.as_vec_basic(), symbolic_cse, opt_level);
    E_hessian_func.init(func_inputs, E_hessian.as_vec_basic(), symbolic_cse, opt_level);
    E_dist_func.init(func_inputs, Dist.as_vec_basic(), symbolic_cse, opt_level);

}






//--------------------------------Help functions-----------------------------------
// For some reason SymEngine doesn't have this implemented X_X
void contactPotentialIMC::subtract_matrix(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C) {
    assert((A.nrows() == B.nrows()) && (A.ncols() == B.ncols()));
    for (unsigned i=0; i < A.nrows(); i++) {
        for (unsigned j=0; j < A.ncols(); j++) {
            C.set(i, j, sub(A.get(i, j), B.get(i, j)));
        }
    }
}

void contactPotentialIMC::get_norm(const DenseMatrix &num, RCP<const Basic> &C) {
    DenseMatrix tmp(num.nrows(), num.ncols());
    num.elementwise_mul_matrix(num, tmp);
    C = sqrt(add(tmp.as_vec_basic()));
}