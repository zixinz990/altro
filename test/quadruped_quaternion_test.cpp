//
// Created by Zixin Zhang on 3/27/23.
// Copyright (c) 2023 Robotic Exploration Lab. All rights reserved.
//

#include <chrono>
#include <filesystem>
#include <iostream>

#include "Eigen/Dense"
#include "altro/altro_solver.hpp"
#include "altro/solver/solver.hpp"
#include "fmt/chrono.h"
#include "fmt/core.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "test_utils.hpp"

using Eigen::MatrixXd;

namespace fs = std::filesystem;
using json = nlohmann::json;

using namespace altro;

TEST(QuadrupedQuatTest, OneMPC) {
  auto t_start = std::chrono::high_resolution_clock::now();
  const int n = QuadrupedQuaternionModel::NumStates;
  const int m = QuadrupedQuaternionModel::NumInputs;
  const int N = 20;
  const float h = 0.01;
  Eigen::Matrix<double, 3, 4> foot_pos_body;
  Eigen::Matrix<double, 3, 3> body_inertia;
  foot_pos_body << 0.17, 0.17, -0.17, -0.17,
                   0.13, -0.13, 0.13, -0.13,
                   -0.3, -0.3, -0.3, -0.3;
  body_inertia << 0.0168128557, 0.0, 0.0,
                  0.0, 0.063009565, 0.0,
                  0.0, 0.0, 0.0716547275;
  double robot_mass = 12.84;
  double body_mass = 5.204;
  Eigen::Matrix3d robot_inertia = (robot_mass / body_mass) * body_inertia;

  ALTROSolver solver(N);

  /// REFERENCES ///
  double contacts[4] = {1.0, 0.0, 0.0, 1.0};
  double num_contacts = 2;

  std::vector<Eigen::VectorXd> X_ref;
  std::vector<Eigen::VectorXd> U_ref;

  for (int i = 0; i <= N; i++) {
    Vector x_ref = Vector::Zero(n);
    Vector u_ref = Vector::Zero(m);

    x_ref << 0.5*0.5*(h*i)*(h*i), 0.0, 0.0,
             1.0, 0.0, 0.0, 0.0,
             0.5*h*i, 0.0, 0.0,
             0.0, 0.0, 0.0;
    u_ref << 0.0, 0.0, contacts[0] * robot_mass * 9.81 / num_contacts,
             0.0, 0.0, contacts[1] * robot_mass * 9.81 / num_contacts,
             0.0, 0.0, contacts[2] * robot_mass * 9.81 / num_contacts,
             0.0, 0.0, contacts[3] * robot_mass * 9.81 / num_contacts;

    X_ref.emplace_back(x_ref);
    U_ref.emplace_back(u_ref);
  }

  /// OBJECTIVE ///
  Eigen::Matrix<double, n, 1> Qd;
  Eigen::Matrix<double, m, 1> Rd;

  double w = 10.0;
  Qd << 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0,  // ignore quaternion in Q
        0.1, 0.1, 0.1,
        0.1, 0.1, 0.1;

  Rd << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6;

  /// DYNAMICS ///
  auto model_ptr = std::make_shared<QuadrupedQuaternionModel>();
  ContinuousDynamicsFunction ct_dyn = [model_ptr, foot_pos_body, robot_inertia, robot_mass](
                                          double *x_dot, const double *x, const double *u) {
    model_ptr->Dynamics(x_dot, x, u, foot_pos_body, robot_inertia, robot_mass);
  };
  ContinuousDynamicsJacobian ct_jac = [model_ptr, foot_pos_body, robot_inertia, robot_mass](
                                          double *jac, const double *x, const double *u) {
    model_ptr->Jacobian(jac, x, u, foot_pos_body, robot_inertia, robot_mass);
  };
  ExplicitDynamicsFunction dt_dyn = ForwardEulerDynamics(n, m, ct_dyn);
  ExplicitDynamicsJacobian dt_jac = ForwardEulerJacobian(n, m, ct_dyn, ct_jac);

  /// CONSTRAINTS ///
  float mu = 0.6;
  float fz_max = 200;
  auto friction_cone_con = [mu, fz_max, contacts](a_float *c, const a_float *x,
                                                          const a_float *u) {
    (void)x;
    for (int i = 0; i < 4; i++) {
      c[0 + i * 6] = u[0 + i * 3] - mu * u[2 + i * 3];      // fx - mu*fz <= 0
      c[1 + i * 6] = -u[0 + i * 3] - mu * u[2 + i * 3];     // -fx - mu*fz <= 0
      c[2 + i * 6] = u[1 + i * 3] - mu * u[2 + i * 3];      // fy - mu*fz <= 0
      c[3 + i * 6] = -u[1 + i * 3] - mu * u[2 + i * 3];     // -fy - mu*fz <= 0
      c[4 + i * 6] = u[2 + i * 3] - fz_max * contacts[i];   // fz <= fz_max
      c[5 + i * 6] = -u[2 + i * 3];  // -fz + 5 <= 0
    }
  };

  // The number of columns of the constraint Jacobian should be n-1+m if using quaternion!
  auto friction_cone_jac = [mu](a_float *jac, const a_float *x, const a_float *u) {
    (void)x;
    (void)u;
    Eigen::Map<Eigen::Matrix<a_float, 24, n - 1 + m>> J(jac);
    J.setZero();

    for (int i = 0; i < 4; i++) {
      J(0 + i * 6, 12 + i * 3) = 1;    // dc0/dfx
      J(0 + i * 6, 14 + i * 3) = -mu;  // dc0/dfz
      J(1 + i * 6, 12 + i * 3) = -1;   // dc1/dfx
      J(1 + i * 6, 14 + i * 3) = -mu;  // dc1/dfz
      J(2 + i * 6, 13 + i * 3) = 1;    // dc2/dfy
      J(2 + i * 6, 14 + i * 3) = -mu;  // dc2/dfz
      J(3 + i * 6, 13 + i * 3) = -1;   // dc3/dfy
      J(3 + i * 6, 14 + i * 3) = -mu;  // dc3/dfz
      J(4 + i * 6, 14 + i * 3) = 1;    // dc4/dfz
      J(5 + i * 6, 14 + i * 3) = -1;   // dc5/dfz
    }
  };

  /// SETUP ///
  AltroOptions opts;
  opts.verbose = Verbosity::Inner;
  opts.iterations_max = 80;
  opts.use_backtracking_linesearch = true;
  opts.use_quaternion = true;
  opts.quat_start_index = 3;  // THIS IS VERY IMPORTANT!
  opts.tol_meritfun_gradient = 1e-8;
  opts.tol_primal_feasibility = 1e-3;
  opts.tol_stationarity = 1e-3;
  solver.SetOptions(opts);

  solver.SetDimension(n, m);
  solver.SetExplicitDynamics(dt_dyn, dt_jac);
  solver.SetTimeStep(h);

  // Cost function
  for (int i = 0; i <= N; i++) {
    solver.SetQuaternionCost(n, m, Qd.data(), Rd.data(), w, X_ref.at(i).data(), U_ref.at(i).data(),
                             i, 0);
  }

  // Try setting constraints for the first 10 knot points
  solver.SetConstraint(friction_cone_con, friction_cone_jac, 24, ConstraintType::INEQUALITY,
                       "friction cone", 0, 10);

  Vector x_init = Vector::Zero(n);
  x_init << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  solver.SetInitialState(x_init.data(), n);
  solver.Initialize();

  // Initial guesses
  for (int i = 0; i <= N; i++) {
    solver.SetState(X_ref.at(i).data(), n, i);
  }
  solver.SetInput(U_ref.at(0).data(), m);

  /// SOLVE ///
  fmt::print("#############################################\n");
  fmt::print("                 MPC Solve\n");
  fmt::print("#############################################\n");

  solver.Solve();
  auto t_end = std::chrono::high_resolution_clock::now();
  using SecondsDouble = std::chrono::duration<double, std::ratio<1>>;
  SecondsDouble t_total = std::chrono::duration_cast<SecondsDouble>(t_end - t_start);
  fmt::print("Total time = {} ms\n", t_total * 1000);

  /// SAVE ///
  std::vector<Vector> X_sim;
  std::vector<Vector> U_sim;

  for (int k = 0; k <= N; k++) {
    Eigen::VectorXd x(n);
    solver.GetState(x.data(), k);
    X_sim.emplace_back(x);
    if (k != N) {
      Eigen::VectorXd u(m);
      solver.GetInput(u.data(), k);
      U_sim.emplace_back(u);
    }
  }

  // Save trajectory to JSON file
  std::filesystem::path test_dir = ALTRO_TEST_DIR;
  std::filesystem::path out_file = test_dir / "quadruped_quaternion_test.json";
  std::ofstream traj_out(out_file);
  json X_ref_data(X_ref);
  json U_ref_data(U_ref);
  json X_data(X_sim);
  json U_data(U_sim);
  json data;
  data["reference_state"] = X_ref_data;
  data["reference_input"] = U_ref_data;
  data["state_trajectory"] = X_data;
  data["input_trajectory"] = U_data;
  traj_out << std::setw(4) << data;
}

TEST(QuadrupedQuatTest, TrotMPC) {
  auto t_start = std::chrono::high_resolution_clock::now();

  const int n = QuadrupedTrotQuaternionModel::NumStates;
  const int m = QuadrupedTrotQuaternionModel::NumInputs;
  const int N = 20;
  const float h = 0.01;

  Eigen::Matrix<double, 3, 2> foot_pos_body;
  Eigen::Matrix<double, 3, 3> body_inertia;
  foot_pos_body << 0.17, -0.17,
                   0.13, -0.13,
                   -0.3, -0.3;
  body_inertia << 0.0168128557, 0.0, 0.0,
                  0.0, 0.063009565, 0.0,
                  0.0, 0.0, 0.0716547275;
  double robot_mass = 12.84;
  double body_mass = 5.204;
  Eigen::Matrix3d robot_inertia = (robot_mass / body_mass) * body_inertia;

  ALTROSolver solver(N);

  /// REFERENCES ///
  double num_contacts = 2;

  std::vector<Eigen::VectorXd> X_ref;
  std::vector<Eigen::VectorXd> U_ref;

  for (int i = 0; i <= N; i++) {
    Vector x_ref = Vector::Zero(n);
    Vector u_ref = Vector::Zero(m);

    x_ref << 0.0, 0.0, 0.0, // position
             1.0, 0.0, 0.0, 0.0,            // quaternion
             0.5*h*i, 0.0, 0.0,             // linear velocity
             0.0, 0.0, 0.0;                 // angular velocity
    x_ref[0] += 0.5*h*i*h;
    u_ref << 0.0, 0.0, robot_mass * 9.81 / num_contacts,
             0.0, 0.0, robot_mass * 9.81 / num_contacts;

    X_ref.emplace_back(x_ref);
    U_ref.emplace_back(u_ref);
  }

  /// OBJECTIVE ///
  Eigen::Matrix<double, n, 1> Qd;
  Eigen::Matrix<double, m, 1> Rd;

  double w = 10.0;
  Qd << 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0,  // ignore quaternion in Q
        10.0, 10.0, 10.0,
        10.0, 10.0, 10.0;

  Rd << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6;

  /// DYNAMICS ///
  auto model_ptr = std::make_shared<QuadrupedTrotQuaternionModel>();
  ContinuousDynamicsFunction ct_dyn = [model_ptr, foot_pos_body, robot_inertia, robot_mass](
                                          double *x_dot, const double *x, const double *u) {
    model_ptr->TrotDynamics(x_dot, x, u, foot_pos_body, robot_inertia, robot_mass);
  };
  ContinuousDynamicsJacobian ct_jac = [model_ptr, foot_pos_body, robot_inertia, robot_mass](
                                          double *jac, const double *x, const double *u) {
    model_ptr->TrotJacobian(jac, x, u, foot_pos_body, robot_inertia, robot_mass);
  };
  ExplicitDynamicsFunction dt_dyn = ForwardEulerDynamics(n, m, ct_dyn);
  ExplicitDynamicsJacobian dt_jac = ForwardEulerJacobian(n, m, ct_dyn, ct_jac);

  /// CONSTRAINTS ///
  float mu = 0.7;
  float fz_max = 200;
  auto friction_cone_con = [mu, fz_max](a_float *c, const a_float *x,
                                                  const a_float *u) {
    (void)x;
    for (int i = 0; i < 2; i++) {
      c[0 + i * 6] = u[0 + i * 3] - mu * u[2 + i * 3];  // fx - mu*fz <= 0
      c[1 + i * 6] = -u[0 + i * 3] - mu * u[2 + i * 3]; // -fx - mu*fz <= 0
      c[2 + i * 6] = u[1 + i * 3] - mu * u[2 + i * 3];  // fy - mu*fz <= 0
      c[3 + i * 6] = -u[1 + i * 3] - mu * u[2 + i * 3]; // -fy - mu*fz <= 0
      c[4 + i * 6] = u[2 + i * 3] - fz_max;             // fz <= fz_max
      c[5 + i * 6] = -u[2 + i * 3];                     // -fz + 5 <= 0
    }
  };

  // The number of columns of the constraint Jacobian should be n-1+m if using quaternion!
  auto friction_cone_jac = [mu](a_float *jac, const a_float *x, const a_float *u) {
    (void)x;
    (void)u;
    Eigen::Map<Eigen::Matrix<a_float, 12, 13 - 1 + 6>> J(jac);
    J.setZero();

    for (int i = 0; i < 2; i++) {
      J(0 + i * 6, 12 + i * 3) = 1;    // dc0/dfx
      J(0 + i * 6, 14 + i * 3) = -mu;  // dc0/dfz
      J(1 + i * 6, 12 + i * 3) = -1;   // dc1/dfx
      J(1 + i * 6, 14 + i * 3) = -mu;  // dc1/dfz
      J(2 + i * 6, 13 + i * 3) = 1;    // dc2/dfy
      J(2 + i * 6, 14 + i * 3) = -mu;  // dc2/dfz
      J(3 + i * 6, 13 + i * 3) = -1;   // dc3/dfy
      J(3 + i * 6, 14 + i * 3) = -mu;  // dc3/dfz
      J(4 + i * 6, 14 + i * 3) = 1;    // dc4/dfz
      J(5 + i * 6, 14 + i * 3) = -1;   // dc5/dfz
    }
  };

  /// SETUP ///
  AltroOptions opts;
  opts.verbose = Verbosity::Inner;
  opts.iterations_max = 30;
  opts.use_backtracking_linesearch = true;
  opts.use_quaternion = true;
  opts.quat_start_index = 3;  // THIS IS VERY IMPORTANT!
//  opts.tol_meritfun_gradient = 1e-8;
//  opts.tol_primal_feasibility = 1e-3;
//  opts.tol_stationarity = 1e-3;
  solver.SetOptions(opts);

  solver.SetDimension(n, m);
  solver.SetExplicitDynamics(dt_dyn, dt_jac);
  solver.SetTimeStep(h);

  // Cost function
  for (int i = 0; i <= N; i++) {
    solver.SetQuaternionCost(n, m, Qd.data(), Rd.data(), w, X_ref.at(i).data(), U_ref.at(i).data(),
                             i, 0);
  }

  // Try setting constraints for the first 10 knot points
  solver.SetConstraint(friction_cone_con, friction_cone_jac, 12, ConstraintType::INEQUALITY,
                       "friction cone", 0, 10);

  Vector x_init = Vector::Zero(n);
  x_init << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  solver.SetInitialState(x_init.data(), n);
  solver.Initialize();

  // Initial guesses
  for (int i = 0; i <= N; i++) {
    solver.SetState(X_ref.at(i).data(), n, i);
  }
  solver.SetInput(U_ref.at(0).data(), m);

  /// SOLVE ///
  fmt::print("#############################################\n");
  fmt::print("                 MPC Solve\n");
  fmt::print("#############################################\n");

  solver.Solve();
  auto t_end = std::chrono::high_resolution_clock::now();
  using SecondsDouble = std::chrono::duration<double, std::ratio<1>>;
  SecondsDouble t_total = std::chrono::duration_cast<SecondsDouble>(t_end - t_start);
  fmt::print("Total time = {} ms\n", t_total * 1000);

  /// SAVE ///
  std::vector<Vector> X_sim;
  std::vector<Vector> U_sim;

  for (int k = 0; k <= N; k++) {
    Eigen::VectorXd x(n);
    solver.GetState(x.data(), k);
    X_sim.emplace_back(x);
    if (k != N) {
      Eigen::VectorXd u(m);
      solver.GetInput(u.data(), k);
      U_sim.emplace_back(u);
    }
  }

  // Save trajectory to JSON file
  std::filesystem::path test_dir = ALTRO_TEST_DIR;
  std::filesystem::path out_file = test_dir / "trot_quadruped_quaternion_test.json";
  std::ofstream traj_out(out_file);
  json X_ref_data(X_ref);
  json U_ref_data(U_ref);
  json X_data(X_sim);
  json U_data(U_sim);
  json data;
  data["reference_state"] = X_ref_data;
  data["reference_input"] = U_ref_data;
  data["state_trajectory"] = X_data;
  data["input_trajectory"] = U_data;
  traj_out << std::setw(4) << data;
}

TEST(QuadrupedQuatTest, EigenTest) {
  Eigen::VectorXd a;

  a.resize(12);
  a << 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4;
  std::cout << a.transpose() << std::endl;

  a.resize(6);
  a << 5, 5, 5, 6, 6, 6;
  std::cout << a.transpose() << std::endl;

  a.resize(12);
  a << 7, 7, 7, 8, 8, 8, 9, 9, 9, 0, 0, 0;
  std::cout << a.transpose() << std::endl;
}