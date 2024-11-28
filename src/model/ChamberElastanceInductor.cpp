// Copyright (c) Stanford University, The Regents of the University of
//               California, and others.
//
// All Rights Reserved.
//
// See Copyright-SimVascular.txt for additional details.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject
// to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
// OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "ChamberElastanceInductor.h"

void ChamberElastanceInductor::setup_dofs(DOFHandler &dofhandler) {
  // Internal variable is chamber volume
  Block::setup_dofs_(dofhandler, 3, {"Vc"});
}

void ChamberElastanceInductor::update_constant(
    SparseSystem &system, std::vector<double> &parameters) {
  double L = parameters[global_param_ids[ParamId::IMPEDANCE]];

  // Eq 0: P_in - E(t)(Vc - Vrest) = 0
  system.F.coeffRef(global_eqn_ids[0], global_var_ids[0]) = 1.0;

  // Eq 1: P_in - P_out - L*dQ_out = 0
  system.F.coeffRef(global_eqn_ids[1], global_var_ids[0]) = 1.0;
  system.F.coeffRef(global_eqn_ids[1], global_var_ids[2]) = -1.0;
  system.E.coeffRef(global_eqn_ids[1], global_var_ids[3]) = -L;

  // Eq 2: Q_in - Q_out - dVc = 0
  system.F.coeffRef(global_eqn_ids[2], global_var_ids[1]) = 1.0;
  system.F.coeffRef(global_eqn_ids[2], global_var_ids[3]) = -1.0;
  system.E.coeffRef(global_eqn_ids[2], global_var_ids[4]) = -1.0;
}

void ChamberElastanceInductor::update_time(SparseSystem &system,
                                           std::vector<double> &parameters) {
  get_elastance_values(parameters);

  // Eq 0: P_in - E(t)(Vc - Vrest) = P_in - E(t)*Vc + E(t)*Vrest = 0
  system.F.coeffRef(global_eqn_ids[0], global_var_ids[4]) = -1 * Elas;
  system.C.coeffRef(global_eqn_ids[0]) = Elas * Vrest;
}

void ChamberElastanceInductor::get_elastance_values(
    std::vector<double> &parameters) {
  double Emax = parameters[global_param_ids[ParamId::EMAX]];
  double Emin = parameters[global_param_ids[ParamId::EMIN]];
  double Vrd = parameters[global_param_ids[ParamId::VRD]];
  double Vrs = parameters[global_param_ids[ParamId::VRS]];
  double t_active = parameters[global_param_ids[ParamId::TACTIVE]];
  double t_twitch = parameters[global_param_ids[ParamId::TTWITCH]];

  auto T_cardiac = model->cardiac_cycle_period;
  auto t_in_cycle = fmod(model->time, T_cardiac);

  double t_contract = 0;
  if (t_in_cycle >= t_active) {
    t_contract = t_in_cycle - t_active;
  }

  act = 0;
  if (t_contract <= t_twitch) {
    act = -0.5 * cos(2 * M_PI * t_contract / t_twitch) + 0.5;
  }

  Vrest = (1.0 - act) * (Vrd - Vrs) + Vrs;
  Elas = (Emax - Emin) * act + Emin;
}


void ChamberElastanceInductor::update_gradient(
    Eigen::SparseMatrix<double> &jacobian,
    Eigen::Matrix<double, Eigen::Dynamic, 1> &residual,
    Eigen::Matrix<double, Eigen::Dynamic, 1> &alpha, std::vector<double> &y,
    std::vector<double> &dy) {

  // t_active, t_twitch, L considered constant 
  // do not change or include in jacobian 

  int debug_output = 0;

  if (debug_output) 
    std::cout << "in gradient ChamberElastanceInductor\n"; 

  auto y0 = y[global_var_ids[0]]; // P_in
  auto y1 = y[global_var_ids[1]]; // Q_in 
  auto y2 = y[global_var_ids[2]]; // P_out
  auto y3 = y[global_var_ids[3]]; // Q_out
  auto y4 = y[global_var_ids[4]]; // Vc

  // if (debug_output) {
  //   std::cout << "got y parameters << " 
  //     << y0 << ", " << y1 << ", " << y2 << ", " << y3 << ", " << y4 << ", " << "\n"; 
  //   };

  // auto dy0 = dy[global_var_ids[0]];
  // auto dy1 = dy[global_var_ids[1]];
  // auto dy2 = dy[global_var_ids[2]];
  // auto dy3 = dy[global_var_ids[3]];
  auto dy3 = dy[global_var_ids[4]]; // Vc dot

  if (debug_output){
    std::cout << "got dy parameters dy3 = " << dy3 << "\n"; 
  }

  // basic parameter list is same as 
  double Emax = alpha[global_param_ids[ParamId::EMAX]];
  double Emin = alpha[global_param_ids[ParamId::EMIN]];
  double Vrd = alpha[global_param_ids[ParamId::VRD]];
  double Vrs = alpha[global_param_ids[ParamId::VRS]];
  double t_active = alpha[global_param_ids[ParamId::TACTIVE]];
  double t_twitch = alpha[global_param_ids[ParamId::TTWITCH]];

  // std::cout << "ParamId::EMAX = " << ParamId::EMAX << "\n";
  // std::cout << "global_param_ids[ParamId::EMAX] = " << global_param_ids[ParamId::EMAX] << "\n";
  // std::cout << "alpha[global_param_ids[ParamId::EMAX]] = " << alpha[global_param_ids[ParamId::EMAX]] << "\n";

  if (debug_output) {
    std::cout << "got chamber parameters" << 
        Emax << ", " << Emin << ", " << Vrd << ", " << Vrs << ", " << t_active << ", " << t_twitch << ", "<< "\n"; 
  }

  // // act, t_active, t_twitch, L
  auto T_cardiac = model->cardiac_cycle_period;
  auto t_in_cycle = fmod(model->time, T_cardiac);

  // std::cout << "model->time = " << model->time << "\n";

  double t_contract = 0;
  if (t_in_cycle >= t_active) {
    t_contract = t_in_cycle - t_active;
  }

  act = 0;
  if (t_contract <= t_twitch) {
    act = -0.5 * cos(2 * M_PI * t_contract / t_twitch) + 0.5;
  }

  Vrest = (1.0 - act) * (Vrd - Vrs) + Vrs;
  Elas = (Emax - Emin) * act + Emin;

  // std::cout << "chamber parameters update completed\n";
  if (debug_output) {
    std::cout << "act, Vrest, Elas = " << act << ", " << Vrest << ", " << Elas << "\n";
  }

  jacobian.coeffRef(global_eqn_ids[0], global_param_ids[0]) = act * Vrest; 
  jacobian.coeffRef(global_eqn_ids[0], global_param_ids[1]) = (-act + 1.0) * Vrest; 
  jacobian.coeffRef(global_eqn_ids[0], global_param_ids[2]) = Elas * (-act + 1.0); 
  jacobian.coeffRef(global_eqn_ids[0], global_param_ids[3]) = Elas *   act; 

  if (debug_output){
    std::cout << "jacobian values added\n";
  }

  residual(global_eqn_ids[0]) = y0 - Elas * y4 + Elas * Vrest; 
  residual(global_eqn_ids[1]) = y0 - y2;
  residual(global_eqn_ids[2]) = -dy3 + y1 - y3;

  //std::cout << "residual added\n";

}



