#include "collisionDetector.h"

collisionDetector::collisionDetector(elasticRod &m_rod, double m_delta = 1e-3, double m_col_limit = 1e-2)
    : rod(&m_rod), delta(m_delta), col_limit(m_col_limit)
{
    contact_limit = delta;
    candidate_limit = col_limit; // 1% of contact limit
    numerical_limit = 0.0; // numerical limit for distance calculations
    cout <<"this class is initialized" << endl;
}

void collisionDetector::updateContactInfo(const MatrixXd &contact_info)
{
    // This function is not implemented yet
    // It should update the contact information based on the current state of the rod
    // and the environment.
    // For now, we will just print a message.
    contact_meshes.clear();

    for (int i = 0; i < contact_info.rows(); i++)
    {
        Vector3d contact_point(contact_info(i, 0), contact_info(i, 1), contact_info(i, 2));
        contact_meshes.push_back(contact_point);
    }
}