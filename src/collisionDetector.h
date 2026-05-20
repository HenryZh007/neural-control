#ifndef COLLISIONDETECTOR_H
#define COLLISIONDETECTOR_H

#include "eigenIncludes.h"
#include "elasticRod.h"


class collisionDetector
{
public:
    collisionDetector(elasticRod &m_rod, double m_delta, double m_col_limit);
    ~collisionDetector() = default;

    bool constructCandidateSet(bool ignore_escape);
    void detectCollisions();
    void updateContactInfo(const MatrixXd &contact_info);


    vector<int> wall_candidate_set;

    int num_collisions;
    double min_dist;

private:
    elasticRod* rod;
    double delta;
    double col_limit;
    
    double contact_limit;
    double candidate_limit;
    double numerical_limit;

    vector<Vector3d> contact_meshes;


    void fixbound(double &x);
    void computeMinDistance(const Vector3d &v1s, const Vector3d &v1e, const Vector3d &v2s, const Vector3d &v2e, double& dist);

};

#endif