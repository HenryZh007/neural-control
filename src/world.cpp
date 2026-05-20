#include "world.h"
#include <chrono>

inline double cross2d(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    return a.x()*b.y() - a.y()*b.x();
}

// Returns {v, theta} where v is center translation, theta (radians) is rotation about the center.
inline std::pair<Eigen::Vector2d, double>
centerShiftAndAngle(const Eigen::Vector2d& p0, const Eigen::Vector2d& p1,
                      const Eigen::Vector2d& q0, const Eigen::Vector2d& q1,
                      double eps = 1e-12)
{
    Eigen::Vector2d c  = 0.5*(p0 + p1);
    Eigen::Vector2d cp = 0.5*(q0 + q1);
    Eigen::Vector2d r  = 0.5*(p0 - p1);
    Eigen::Vector2d rp = 0.5*(q0 - q1);

    if (r.squaredNorm() < eps || rp.squaredNorm() < eps)
        throw std::runtime_error("Degenerate segment (length ~ 0).");

    double theta = std::atan2(cross2d(r, rp), r.dot(rp));  // (-pi, pi]
    Eigen::Vector2d v = cp - c;                            // center shift

    return {v, theta};
}



Eigen::MatrixXd loadMatrixFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        double value;

        while (iss >> value) {
            row.push_back(value);
        }

        if (row.size() != 3) {
            throw std::runtime_error("Invalid row size in file. Expected 3 columns.");
        }

        data.push_back(row);
    }

    file.close();

    // Convert vector of vectors to Eigen::MatrixXd
    Eigen::MatrixXd matrix(data.size(), 3);
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            matrix(i, j) = data[i][j];
        }
    }

    return matrix;
}

world::world(setInput &m_inputData)
{
	render = m_inputData.GetBoolOpt("render");				// boolean
	saveData = m_inputData.GetBoolOpt("saveData");			// boolean

	// Physical parameters
	RodLength = m_inputData.GetScalarOpt("RodLength");      // meter
    gVector = m_inputData.GetVecOpt("gVector");             // m/s^2
    maxIter = m_inputData.GetIntOpt("maxIter");             // maximum number of iterations
	rodRadius = m_inputData.GetScalarOpt("rodRadius");      // meter
	numVertices = m_inputData.GetIntOpt("numVertices");     // int_num
	youngM = m_inputData.GetScalarOpt("youngM");            // Pa
	Poisson = m_inputData.GetScalarOpt("Poisson");          // dimensionless
	deltaTime = m_inputData.GetScalarOpt("deltaTime");      // seconds
	totalTime= m_inputData.GetScalarOpt("totalTime");       // seconds
	tol = m_inputData.GetScalarOpt("tol");                  // small number like 10e-7
	stol = m_inputData.GetScalarOpt("stol");				// small number, e.g. 0.1%
	density = m_inputData.GetScalarOpt("density");          // kg/m^3
	viscosity = m_inputData.GetScalarOpt("viscosity");      // viscosity in Pa-s

	shearM = youngM/(2.0*(1.0+Poisson));					// shear modulus

	render = 1.0;


	opts.Delta0 = 1.0;
	opts.eta = 0.1;
	opts.tol_g = 1e-9;
	opts.shrink = 0.25;
	opts.expand = 2.0;

}

world::~world()
{
	cleanup();
}

bool world::isRender()
{
	return render;
}

void world::OpenFile(ofstream &outfile)
{
	if (saveData==false) return;

	int systemRet = system("mkdir datafiles"); //make the directory
	if(systemRet == -1)
	{
		cout << "Error in creating directory\n";
	}

	time_t current_time = time(0);

	// Open an input file named after the current time
	ostringstream name;
    name << "datafiles/simDER.txt";

	outfile.open(name.str().c_str());
	outfile.precision(10);
}

void world::CloseFile(ofstream &outfile)
{
	if (saveData==false)
		return;

	outfile.close();
}

void world::CoutData(ofstream &outfile)
{
	if (saveData==false)
		return;

	if (currentTime < 100) return;

	for (int i = 0; i < rod->nv; i++)
	{
		Vector3d xCurrent = rod->getVertex(i);
		outfile << xCurrent(0) << " " 
				<< xCurrent(1) << " "
				<< xCurrent(2) << endl;
	}

	currentTime = totalTime;


}


void world::resetSim()
{
	// reset the simulation
	cleanup();

	rodGeometry(); // reset the geometry

	// rod = new elasticRod(vertices, vertices0, density, rodRadius, deltaTime,
	// 	youngM, shearM, RodLength, theta);
	rod = make_shared<elasticRod>(vertices, vertices0, density, rodRadius, deltaTime,
		youngM, shearM, RodLength, theta);

	characteristicForce = M_PI * pow(rodRadius ,4)/4.0 * youngM / pow(RodLength, 2);
	forceTol = tol * characteristicForce;

	rodBoundaryCondition(); // reset the boundary condition
	for (int i = 0; i < control_info.rows(); i++)
	{
		int nodeIndex = control_info(i, 1);
		if (nodeIndex < 0 || nodeIndex >= rod->nv)
		{
			cout << "Error: node index out of bounds." << endl;
			exit(1);
		}

		if (control_info(i, 0) == 0)
		{
			Vector3d xCurrent = rod->getVertex(nodeIndex);
			rod->setVertexBoundaryCondition(xCurrent, nodeIndex);
		}
	}

	// setup the rod so that all the relevant variables are populated
	rod->setup();
	// End of rod setup

	// set up the time stepper
	stepper = make_shared<timeStepper>(*rod);

	// declare the forces
	m_stretchForce = make_unique<elasticStretchingForce>(*rod, *stepper);
	m_bendingForce = make_unique<elasticBendingForce>(*rod, *stepper);
	m_twistingForce = make_unique<elasticTwistingForce>(*rod, *stepper);
	m_inertialForce = make_unique<inertialForce>(*rod, *stepper);
	m_gravityForce = make_unique<externalGravityForce>(*rod, *stepper, gVector);
	m_dampingForce = make_unique<dampingForce>(*rod, *stepper, viscosity);
	m_contactPotentialIMC = make_unique<contactPotentialIMC>(*rod, *stepper, d_h, col_limit, k_scaler);	
	m_contactPotentialIMC->updateContactInfo(contact_info);
	m_bendingCompute = make_unique<bendingCompute>(*rod, *stepper);

	tr_solver = make_shared<solver>(*rod, *stepper, *m_stretchForce, *m_bendingForce,
								    *m_gravityForce);




	Nstep = totalTime/deltaTime;

	// Allocate every thing to prepare for the first iteration
	rod->updateTimeStep();

	timeStep = 0;
	currentTime = 0.0;
}



void world::setRodStepper()
{
	// Set up geometry
	rodGeometry();

	// Create the rod	
	rod = make_shared<elasticRod>(vertices, vertices0, density, rodRadius, deltaTime,
		youngM, shearM, RodLength, theta);


	// Find out the tolerance, e.g. how small is enough?
	characteristicForce = M_PI * pow(rodRadius ,4)/4.0 * youngM / pow(RodLength, 2);
	forceTol = tol * characteristicForce;

	// Set up boundary condition
	rodBoundaryCondition();

	// setup the rod so that all the relevant variables are populated
	rod->setup();
	// End of rod setup

	// set up the time stepper
	stepper = make_shared<timeStepper>(*rod);

	// // declare the forces
	m_stretchForce = make_unique<elasticStretchingForce>(*rod, *stepper);
	m_bendingForce = make_unique<elasticBendingForce>(*rod, *stepper);
	m_twistingForce = make_unique<elasticTwistingForce>(*rod, *stepper);
	m_inertialForce = make_unique<inertialForce>(*rod, *stepper);
	m_gravityForce = make_unique<externalGravityForce>(*rod, *stepper, gVector);
	m_dampingForce = make_unique<dampingForce>(*rod, *stepper, viscosity);
	m_contactPotentialIMC = make_unique<contactPotentialIMC>(*rod, *stepper, d_h, col_limit, k_scaler);
	
	m_bendingCompute = make_unique<bendingCompute>(*rod, *stepper);

	Nstep = totalTime/deltaTime;

	// Allocate every thing to prepare for the first iteration
	rod->updateTimeStep();

	timeStep = 0;
	currentTime = 0.0;
}

// Setup geometry
void world::rodGeometry()
{
	vertices = loadMatrixFromFile("inputs/" + fileName);

	numVertices = vertices.rows();

	// compute the RodLength
	RodLength = 1.0;

	vertices0 = MatrixXd(numVertices, 3);
    double delta_l = RodLength / (numVertices - 1);

    for (int i = 0; i < numVertices; i ++)
    {
        vertices0(i,0) = i * delta_l;
        vertices0(i,1) = 0;
        vertices0(i,2) = 0;
    }

    // initial theta should be zeros
    theta = VectorXd::Zero(numVertices - 1);
}

void world::rodBoundaryCondition()
{
	// Apply boundary condition
	// rod->setVertexBoundaryCondition(rod->getVertex(0),0);
	// rod->setThetaBoundaryCondition(rod->getTheta(0),0);
	// rod->setVertexBoundaryCondition(rod->getVertex(1),1);

	// rod->setVertexBoundaryCondition(rod->getVertex(rod->nv-1),rod->nv-1);
	// rod->setThetaBoundaryCondition(rod->getTheta(rod->ne-1),0);
	// rod->setVertexBoundaryCondition(rod->getVertex(rod->nv-2),rod->nv-2);

	for (int i = 0; i < rod->ne; i++)
	{
		rod->setThetaBoundaryCondition(rod->getTheta(i), i);
		// constrained y
		rod->setDOFBoundaryCondition(0.0, 4*i + 1); // y coordinate
	}
}

void world::setXb(const Eigen::MatrixXd &xb)
{

	x_t_0 = Vector3d(xb(0), 0, 0);
	x_t_n = Vector3d(xb(1), 0, 0);
}

// void world::updateControlInputs(const Eigen::MatrixXd &m_control_inputs)
// {
// 	control_inputs.resize(m_control_inputs.rows(), 3);
// 	for (int i = 0; i < control_inputs.rows(); i++)
// 	{
// 		control_inputs(i, 0) = m_control_inputs(i, 0);
// 		control_inputs(i, 1) = 0;
// 		control_inputs(i, 2) = m_control_inputs(i, 1);
// 	}
// }

void world::updateControlInputs(const Eigen::MatrixXd &m_control_inputs)
{
	control_inputs = m_control_inputs;

	// cout << control_inputs << endl;

	for (int i = 0; i < control_info.rows(); i++)
	{
		if (control_info(i, 0) == 0) // position control
		{
			Vector3d x_control = Vector3d(m_control_inputs(i, 0), 0, m_control_inputs(i, 1));
			rod->setVertexBoundaryCondition(x_control, control_info(i, 1));
		}
	}
}

double world::getVelocity()
{
	return rod->u.norm();
}


void world::defineController(const Eigen::MatrixXi &control_inputs)
{
	control_info = control_inputs;

	// update the boundary condition
	if (control_info.cols() != 2)
	{
		cout << "Error: control inputs should be a (N, 2) matrix." << endl;
		exit(1);
	}

	for (int i = 0; i < control_info.rows(); i++)
	{
		int nodeIndex = control_info(i, 1);
		if (nodeIndex < 0 || nodeIndex >= rod->nv)
		{
			cout << "Error: node index out of bounds." << endl;
			exit(1);
		}

		if (control_info(i, 0) == 0)
		{
			Vector3d xCurrent = rod->getVertex(nodeIndex);
			rod->setVertexBoundaryCondition(xCurrent, nodeIndex);
		}
	}
	rod->updateMap();
	stepper->update();

}



void world::updateBoundaryCondition()
{
	// Update the boundary condition
	Vector3d x_nm1 = rod->getVertex(rod->nv-2);
	Vector3d x_n = rod->getVertex(rod->nv-1);

	if (abs(x_n(0) - 0.5) > 0.01 * deltaTime)
	{
		x_nm1(0) -= 0.01 * deltaTime; // move the last node a little bit
		x_n(0) -= 0.01 * deltaTime;

		rod->setVertexBoundaryCondition(x_nm1, rod->nv-2);
		rod->setVertexBoundaryCondition(x_n, rod->nv-1);
	}
	else{
		x_n(0) = 0.5; // set the last node to be at 0.5
		rod->setVertexBoundaryCondition(x_n, rod->nv-1);
		rod->setVertexBoundaryCondition(x_nm1, rod->nv-2);
	}

	
	if (x_n(0) == 0.5)
	{
		x_nm1 = x_n;
		x_nm1(2) += 0.01;
		rod->setVertexBoundaryCondition(x_nm1, rod->nv-2);

		Vector3d x0 = rod->getVertex(0);
		Vector3d x1 = rod->getVertex(1);

		x1 = x0;
		x1(2) += 0.01;
		rod->setVertexBoundaryCondition(x1, 1);


	
		// t1 = t1.normalized();
		// double deltaAngle = M_PI / 180;

		// double cosTheta = cos(deltaAngle);
		// double sinTheta = sin(deltaAngle);

		// Matrix3d rotationMatrix;
		// rotationMatrix << cosTheta, 0, sinTheta,
		// 				  0, 1, 0,
		// 				  -sinTheta, 0, cosTheta;

		// t1 = rotationMatrix * t1; // rotate the tangent vector
		// t1.normalize();

		// cout << t1 << endl;
		// exit(0);

	}
}


void world::updateTowardsTarget(MatrixXd &current_status, double max_step_norm){
	for (int i = 0; i < current_status.rows(); ++i){
		VectorXd diff = control_inputs.row(i) - current_status.row(i);
		double norm = diff.norm();

		if (norm > max_step_norm * rod->dt){
			diff *= (max_step_norm * rod->dt/norm);
		}
		current_status.row(i) += diff;
	}	
}

bool world::hasConverged(const MatrixXd &current_status){
	for (int i = 0; i < current_status.rows(); ++i){
		VectorXd diff = control_inputs.row(i) - current_status.row(i);
		if (diff.norm() > 1e-2 * rod->dt){
			return false;
		}
	}
	return true;
}

void world::getControlStatus(MatrixXd &current_status)
{
	current_status.resize(control_info.rows(), 3);
	for (int i = 0; i < control_info.rows(); i++)
	{
		int nodeIndex = control_info(i, 1);
		if (nodeIndex < 0 || nodeIndex >= rod->nv)
		{
			cout << "Error: node index out of bounds." << endl;
			exit(1);
		}

		Vector3d xCurrent = rod->getVertex(nodeIndex);
		current_status.row(i) = xCurrent.transpose();
	}
}

void world::updateControlStatus(const MatrixXd &control_status)
{
	for (int i = 0; i < control_info.rows(); i++)
	{
		int nodeIndex = control_info(i, 1);
		Vector3d xCurrent = control_status.row(i);
		rod->setVertexBoundaryCondition(xCurrent, nodeIndex);
	}

}


// bool world::computeEqulibrium()
// {
// 	bool solved = false;
// 	double epsilon = 1.0; // initial epsilon

// 	m_inertialForce->updateEpsilon(0.0); // reset epsilon to 1.0

// 	// verify the accuracy

// 	Vector2d t1_l = Vector2d(0, 0);
// 	double w1_l = 0.0;

// 	Vector2d t2_l = Vector2d(0, 0);
// 	double w2_l = 0.0;

// 	Vector2d t1_t = target_translation[0];
// 	double w1_t = target_rotation[0];

// 	Vector2d t2_t = target_translation[1];
// 	double w2_t = target_rotation[1];

// 	Vector2d t1_r = t1_t; 
// 	double w1_r = w1_t;

// 	Vector2d t2_r = t2_t; 
// 	double w2_r = w2_t;

// 	Vector2d x0 = Vector2d(rod->getVertex(0)(0), rod->getVertex(0)(2));
// 	Vector2d x1 = Vector2d(rod->getVertex(1)(0), rod->getVertex(1)(2));
// 	Vector2d xnm1 = Vector2d(rod->getVertex(rod->nv-2)(0), rod->getVertex(rod->nv-2)(2));
// 	Vector2d xn = Vector2d(rod->getVertex(rod->nv-1)(0), rod->getVertex(rod->nv-1)(2));

// 	int inner_iter = 0;

// 	VectorXd xold;
// 	while (true)
// 	{
// 		Rotation2Dd R1(w1_r);
// 		Vector2d c1 = 0.5 * (x0 + x1);
// 		Vector2d cp1 = c1 + t1_r;
		
// 		Vector2d x0_t = cp1 + R1 * (x0 - c1);
// 		Vector2d x1_t = cp1 + R1 * (x1 - c1);	

// 		Rotation2Dd R2(w2_r);
// 		Vector2d c2 = 0.5 * (xnm1 + xn);
// 		Vector2d cp2 = c2 + t2_r;
		
// 		Vector2d xnm1_t = cp2 + R2 * (xnm1 - c2);
// 		Vector2d xn_t = cp2 + R2 * (xn - c2);

// 		rod->setVertexBoundaryCondition(Vector3d(x0_t(0), 0, x0_t(1)), 0);
// 		rod->setVertexBoundaryCondition(Vector3d(x1_t(0), 0, x1_t(1)), 1);
// 		rod->setVertexBoundaryCondition(Vector3d(xn_t(0), 0, xn_t(1)), rod->nv-1);
// 		rod->setVertexBoundaryCondition(Vector3d(xnm1_t(0), 0, xnm1_t(1)), rod->nv-2);

// 		solved = false;
// 		xold = rod->x;
// 		newtonMethod(solved); // try to solve the equilibrium state

// 		// cout <<"inner: " << inner_iter  <<" " << t1_r.transpose() << " " << t1_t.transpose() <<
// 		// " " << solved<<  endl;
// 		// cout << t2_r.transpose() << " " << t2_t.transpose() << endl;

// 		if (inner_iter > 100)
// 		{
// 			exit(0);
// 		}

// 		if (solved)
// 		{
// 			if ((t1_r - t1_t).norm() < 1e-5 && (t2_r - t2_t).norm() < 1e-5)
// 			{
// 				break;	
// 			}
// 			else{
// 				t1_r = t1_t;
// 				w1_r = w1_t;
// 				t2_r = t2_t;
// 				w2_r = w2_t;
// 			}
// 		}
// 		else{
// 			t1_r = (t1_r + t1_l) * 0.5;
// 			w1_r = (w1_r + w1_l) * 0.5;
// 			t2_r = (t2_r + t2_l) * 0.5;
// 			w2_r = (w2_r + w2_l) * 0.5;
// 			rod->x = xold;
// 		}


// 		inner_iter++;
// 	}
// 	cout <<"inner_iter: " << inner_iter << endl;


// 	return true;
// }

bool world::computeEqulibrium()
{
	bool solved = false;
	double epsilon = 1.0; // initial epsilon

	m_inertialForce->updateEpsilon(1.0); // reset epsilon to 1.0

	// verify the accuracy

	Vector2d t1_l = Vector2d(0, 0);
	double w1_l = 0.0;

	Vector2d t2_l = Vector2d(0, 0);
	double w2_l = 0.0;

	Vector2d t1_r = target_translation[0];
	double w1_r = target_rotation[0];

	Vector2d t2_r = target_translation[1];
	double w2_r = target_rotation[1];

	// Vector2d t1_v = (t1_r - t1_l) / 10.0;
	// Vector2d t2_v = (t2_r - t2_l) / 10.0;
	// double w1_v = (w1_r - w1_l) / 10.0;
	// double w2_v = (w2_r - w2_l) / 10.0;


	Vector2d x0 = Vector2d(rod->getVertex(0)(0), rod->getVertex(0)(2));
	Vector2d x1 = Vector2d(rod->getVertex(1)(0), rod->getVertex(1)(2));
	Vector2d xnm1 = Vector2d(rod->getVertex(rod->nv-2)(0), rod->getVertex(rod->nv-2)(2));
	Vector2d xn = Vector2d(rod->getVertex(rod->nv-1)(0), rod->getVertex(rod->nv-1)(2));

	int inner_iter = 0;

	for (int i = 1; i <= 10; i++)
	{
		double alpha = i * 1.0 / 10.0;
		Vector2d t1_c = t1_l * (1.0 -alpha) + t1_r * alpha;
		double w1_c = w1_l * (1.0 - alpha) + w1_r * alpha;
		Vector2d t2_c = t2_l * (1.0 -alpha) + t2_r * alpha;
		double w2_c = w2_l * (1.0 - alpha) + w2_r * alpha;

		Rotation2Dd R1(w1_c);
		Vector2d c1 = 0.5 * (x0 + x1);
		Vector2d cp1 = c1 + t1_c;
		
		Vector2d x0_t = cp1 + R1 * (x0 - c1);
		Vector2d x1_t = cp1 + R1 * (x1 - c1);	

		Rotation2Dd R2(w2_c);
		Vector2d c2 = 0.5 * (xnm1 + xn);
		Vector2d cp2 = c2 + t2_c;
		
		Vector2d xnm1_t = cp2 + R2 * (xnm1 - c2);
		Vector2d xn_t = cp2 + R2 * (xn - c2);

		rod->setVertexBoundaryCondition(Vector3d(x0_t(0), 0, x0_t(1)), 0);
		rod->setVertexBoundaryCondition(Vector3d(x1_t(0), 0, x1_t(1)), 1);
		rod->setVertexBoundaryCondition(Vector3d(xn_t(0), 0, xn_t(1)), rod->nv-1);
		rod->setVertexBoundaryCondition(Vector3d(xnm1_t(0), 0, xnm1_t(1)), rod->nv-2);

		solved = false;
		newtonMethod(solved); // try to solve the equilibrium state

		if (!solved)
		{
			cout << "Error: Could not converge with alpha: " << alpha << endl;
			cout << "t1_c: " << t1_c.transpose() << ", w1_c: " << w1_c << endl;
			cout << "t2_c: " << t2_c.transpose() << ", w2_c: " << w2_c << endl;
			exit(1);
		}
	}

	// check the difference 
	Vector3d x0_c = rod->getVertex(0);
	Vector3d x1_t = Vector3d(control_inputs(0, 0), 0, control_inputs(0, 1));

	// cout <<"diff: " << (x0_c - x1_t).norm() << endl;



	// cout << rod->getVertex(0).transpose() << endl;
	// cout << rod->getVertex(1).transpose() << endl;
	// cout << rod->getVertex(rod->nv-2).transpose() << endl;
	// cout << rod->getVertex(rod->nv-1).transpose() << endl;
	// cout << control_inputs << endl;

	// exit(0);

	m_inertialForce->updateEpsilon(1.0); // reset epsilon to 1.0


	while (rod->u.norm() > 1e-3)
	{
		solved = false;
		newtonMethod(solved); // update
		// cout <<"velocity: " << rod->u.norm() << " solved: " << solved << endl;

		if (!solved)
		{
			cout << "Error: Could not converge to static equilibrium." << endl;
			exit(1);
		}
	}


	// VectorXd xold;
	// while (true)
	// {
	// 	Rotation2Dd R1(w1_r);
	// 	Vector2d c1 = 0.5 * (x0 + x1);
	// 	Vector2d cp1 = c1 + t1_r;
		
	// 	Vector2d x0_t = cp1 + R1 * (x0 - c1);
	// 	Vector2d x1_t = cp1 + R1 * (x1 - c1);	

	// 	Rotation2Dd R2(w2_r);
	// 	Vector2d c2 = 0.5 * (xnm1 + xn);
	// 	Vector2d cp2 = c2 + t2_r;
		
	// 	Vector2d xnm1_t = cp2 + R2 * (xnm1 - c2);
	// 	Vector2d xn_t = cp2 + R2 * (xn - c2);

	// 	rod->setVertexBoundaryCondition(Vector3d(x0_t(0), 0, x0_t(1)), 0);
	// 	rod->setVertexBoundaryCondition(Vector3d(x1_t(0), 0, x1_t(1)), 1);
	// 	rod->setVertexBoundaryCondition(Vector3d(xn_t(0), 0, xn_t(1)), rod->nv-1);
	// 	rod->setVertexBoundaryCondition(Vector3d(xnm1_t(0), 0, xnm1_t(1)), rod->nv-2);

	// 	solved = false;
	// 	xold = rod->x;
	// 	newtonMethod(solved); // try to solve the equilibrium state

	// 	// cout <<"inner: " << inner_iter  <<" " << t1_r.transpose() << " " << t1_t.transpose() <<
	// 	// " " << solved<<  endl;
	// 	// cout << t2_r.transpose() << " " << t2_t.transpose() << endl;

	// 	if (inner_iter > 100)
	// 	{
	// 		exit(0);
	// 	}

	// 	if (solved)
	// 	{
	// 		if ((t1_r - t1_t).norm() < 1e-5 && (t2_r - t2_t).norm() < 1e-5)
	// 		{
	// 			break;	
	// 		}
	// 		else{
	// 			t1_r = t1_t;
	// 			w1_r = w1_t;
	// 			t2_r = t2_t;
	// 			w2_r = w2_t;
	// 		}
	// 	}
	// 	else{
	// 		t1_r = (t1_r + t1_l) * 0.5;
	// 		w1_r = (w1_r + w1_l) * 0.5;
	// 		t2_r = (t2_r + t2_l) * 0.5;
	// 		w2_r = (w2_r + w2_l) * 0.5;
	// 		rod->x = xold;
	// 	}


	// 	inner_iter++;
	// }
	// cout <<"inner_iter: " << inner_iter << endl;


	return true;
}




// bool world::computeEqulibrium()
// {
// 	bool solved = false;
// 	double epsilon = 1.0; // initial epsilon

// 	m_inertialForce->updateEpsilon(1.0); // reset epsilon to 1.0

// 	MatrixXd control_status;

// 	while (true)
// 	{
// 		// get the control_status
// 		getControlStatus(control_status);		

// 		updateTowardsTarget(control_status);
// 		updateControlStatus(control_status);
// 		// update the boundary condition

// 		solved = false;
// 		newtonMethod(solved); // update
// 		if (!solved)
// 		{
// 			cout << control_status << endl;
// 		 	break;
// 		}
		
// 		if (hasConverged(control_status))
// 		{
// 			break;
// 		}
// 	}

// 	if (!solved){
// 		return false;
// 	}



// 	epsilon = 1.0;
// 	VectorXd xold = rod->x;
// 	for (int i = 0; i < 100; i++)
// 	{
// 		epsilon -= 1.0/100;
// 		if (epsilon < 0)
// 		{
// 			epsilon = 0.0; // reset epsilon to 0.0
// 		}
// 		m_inertialForce->updateEpsilon(epsilon); // update the epsilon
// 		solved = false;
// 		newtonMethod(solved); // update
// 		if (!solved)
// 		{
// 			cout << "Error: Could not converge with epsilon: " << epsilon << endl;
// 			break;
// 		}
// 		xold = rod->x;
// 	}
// 	if (!solved)
// 	{
// 		for (int i = 0; i < 10; i++)
// 		{
// 			m_inertialForce->updateEpsilon(epsilon + 1.0/ 100);
// 			solved = false;
// 			newtonMethod(solved); // update
// 		}
// 	}

// 	return true;
// }

void world::newtonMethod(bool &solved)
{
	double normf = forceTol * 10.0;
	double normf0 = 0;

	iter = 0;

	// Start with a trial solution for our solution x
	rod->updateGuess(); // x = x0 + u * dt
	// m_contactPotentialIMC->constructCandidateSet();
	while (solved == false)
	{
		rod->prepareForIteration();

		stepper->setZero();

		m_inertialForce->computeFi();
		m_inertialForce->computeJi();

		m_stretchForce->computeFs();
		m_stretchForce->computeJs();

		m_bendingForce->computeFb();
		m_bendingForce->computeJb();

		// m_twistingForce->computeFt();
		// m_twistingForce->computeJt();

		// m_gravityForce->computeFg();
		// m_gravityForce->computeJg();

		m_dampingForce->computeFd();
		m_dampingForce->computeJd();

		// m_contactPotentialIMC->computeFcJc();

		normf = stepper->force.norm();

		// cout <<"iter: " << iter << " normf: " << normf << " maxIter: " << maxIter <<
		// " " << forceTol << " " << normf * stol << endl;

		// check if the value is nan
		if (isnan(normf) || isinf(normf))
		{
			cout <<"Error : normf is nan or inf. Exiting." << endl;
			solved = false;
			break;
		}

		if (iter == 0)
		{
			normf0 = normf;
		}

		if (normf <= forceTol)
		{
			solved = true;
		}
		else if(iter > 0 && normf <= 1e-9)
		{
			solved = true;
		}

		if (solved == false)
		{
			stepper->integrator(); // Solve equations of motion

			// double alpha = lineSearch();
			double alpha = 1.0;
			// cout <<"alpha: " << alpha << endl;
			
			rod->updateNewtonX(stepper->dx.data(), alpha); // new q = old q + Delta q
			iter++;
		}

		if (iter > maxIter)
		{
			cout <<"iter: " << iter << " normf: " << normf << endl;
			// // solved = true;
			// cout << "Error. Could not converge. Exiting.\n";
			break;
		}
	}
}


double world::lineSearch()
{

	double fx = 0.5 * stepper->force.norm(); 


	double alpha = 1.0;
	VectorXd xold = rod->x; // save the old x

	for (int i = 0; i < 100; i++)
	{
		rod->x = xold;
		rod->updateNewtonX(stepper->dx.data(), alpha); // update the position		
		rod->prepareForIteration();

		stepper->setZero();
		// m_inertialForce->computeFi();
		m_stretchForce->computeFs();
		m_bendingForce->computeFb();
		// m_twistingForce->computeFt();
		// m_gravityForce->computeFg();
		m_dampingForce->computeFd();
		// m_contactPotentialIMC->computeFcJc();


		double fx_new = 0.5 * stepper->force.norm();

		if (fx_new <= fx)
		{
			break;
		}
		alpha *= 0.5;
		if (alpha < 1e-5)
		{
			// cout <<"Alpha become too small: " << stepper->dx.norm() << endl;
			// exit(0);
			break;
		}
	}

	rod->x = xold;

	return alpha;

}






void world::updateTimeStep()
{

	// update boundary
	// updateBoundaryCondition();
	// bool flag =  computeEqulibrium();
	for (int i = 0; i < 2; i ++)
	{
		VectorXd x0 = rod->getFreeDOF();
		VectorXd x = tr_solver->minimizeEnergy_TR(x0, opts);
		// set the free DOF
		rod->setFreeDOF(x);


		rod->updateTimeStep(); // update the time step
	} 

	currentTime += deltaTime;
	timeStep++;
}

void world::updateTimeStepWithInertia()
{
	bool solved = false;
	newtonMethod(solved);

	rod->updateTimeStep(); // update the time step
}



bool world::simulationRunning()
{
	if (currentTime < totalTime)
		return true;
	else
	{
		return false;
	}
}

int world::numPoints()
{
	return rod->nv;
}

double world::getScaledCoordinate(int i)
{
	return rod->x[i] / RodLength;
}

double world::getCurrentTime()
{
	return currentTime;
}

double world::getTotalTime()
{
	return totalTime;
}

MatrixXd world::getAllCoordinates()
{
	MatrixXd coordinates(rod->nv, 2);
	for (int i = 0; i < rod->nv; i++)
	{
		Vector3d xCurrent = rod->getVertex(i);
		coordinates(i, 0) = xCurrent(0);
		coordinates(i, 1) = xCurrent(2);
	}
	return coordinates;
}


VectorXd world::getForce()
{

	rod->prepareForIteration();
	stepper->setZero();
 	m_stretchForce->computeFs();
 	m_bendingForce->computeFb();

	return stepper->getForce_py();
}

VectorXd world::getStretchForce()
{
	m_inertialForce->updateEpsilon(0.0); // reset epsilon to 0.0
	rod->prepareForIteration();
	stepper->setZero();
 	m_stretchForce->computeFs();

	return stepper->getForce_py();
}

MatrixXd world::getJacobian()
{
	m_inertialForce->updateEpsilon(0.0); // reset epsilon to 0.0

	rod->prepareForIteration();
	stepper->setZero();
 	m_stretchForce->computeJs();
 	m_bendingForce->computeJb();
 	// m_gravityForce->computeJg();

	return stepper->getJacobian_py();
}


// VectorXd world::computedKap(const Eigen::MatrixXd &kappa_bar)
// {
// 	return m_bendingCompute->computeGrad(kappa_bar);
// }

MatrixXd world::computeCurvature(const Eigen::MatrixXd &vertices)
{
	return rod->computeCurvature(vertices);
}



void world::updateContactInfo(const Eigen::MatrixXd &m_contact_info)
{
	contact_info = m_contact_info;
	m_contactPotentialIMC->updateContactInfo(contact_info);
}


void world::cleanup()
{
	if (rod) rod.reset(); // reset the rod
	if (stepper) stepper.reset(); // reset the stepper
	if (m_stretchForce) m_stretchForce.reset(); // reset the stretch force
	if (m_bendingForce) m_bendingForce.reset(); // reset the bending force
	if (m_twistingForce) m_twistingForce.reset(); // reset the twisting force
	if (m_inertialForce) m_inertialForce.reset(); // reset the inertial force
	if (m_gravityForce) m_gravityForce.reset(); // reset the gravity force
	if (m_dampingForce) m_dampingForce.reset(); // reset the damping force
	if (m_contactPotentialIMC) m_contactPotentialIMC.reset(); // reset the contact potential
	if (m_bendingCompute) m_bendingCompute.reset(); // reset the bending compute
}

double world::computeCurvatureLoss(const Eigen::MatrixXd &kappa_bar, const Eigen::Matrix2d *coeff)
{
	return m_bendingCompute->computeLoss(kappa_bar, coeff);
}

VectorXd world::computeDCurvature(const MatrixXd &kappa_bar, const Eigen::Matrix2d *coeff)
{
	return m_bendingCompute->computeGrad(kappa_bar, coeff);
}

double world::computeStretchLoss(std::optional<double> coeff)
{
	return m_bendingCompute->computeStretchLoss(coeff);
}

VectorXd world::computeStretchGrad(std::optional<double> coeff)
{
	return m_bendingCompute->computeStretchGrad(coeff);
}

MatrixXd world::getAllFrames()
{
	return rod->m1;
}



void world::setAllFrames(const Eigen::MatrixXd &m1)
{
	rod->computeTangent(rod->x, rod->tangent);
	rod->m1 = m1;
	for (int i = 0; i < rod->ne; i++)
	{
		Vector3d m1_local = rod->m1.row(i);
		Vector3d t_local = rod->tangent.row(i);
		rod->m2.row(i) = t_local.cross(m1_local);
	}
	rod->d1 = rod->m1;
	rod->d2 = rod->m2;
}

void world::setAllVertices(const Eigen::VectorXd &X)
{
	for (int i = 0; i < rod->nv; i++)
	{
		rod->x(4*i) = X(2*i);
		rod->x(4*i + 2) = X(2*i + 1);
	}

	rod->x0 = rod->x;
	rod->u = VectorXd::Zero(rod->ndof);
}