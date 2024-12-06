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

  // cos act function specific parameters 
  double t_active = parameters[global_param_ids[ParamId::TACTIVE]];
  double t_twitch = parameters[global_param_ids[ParamId::TTWITCH]];

  // two hill specific parameters 
  double t_shift = parameters[global_param_ids[ParamId::TSHIFT]];
  double tau_1 = parameters[global_param_ids[ParamId::TAU_1]];
  double tau_2 = parameters[global_param_ids[ParamId::TAU_2]];
  double m1 = parameters[global_param_ids[ParamId::M1]];
  double m2 = parameters[global_param_ids[ParamId::M2]];

  // switch parameter 
  auto two_hill = parameters[global_param_ids[ParamId::TWO_HILL]];


  auto T_cardiac = model->cardiac_cycle_period;
  auto t_in_cycle = fmod(model->time, T_cardiac);

  double act = 0;
  double act_two_hill = 0;
  
  static bool first_entry = true;

  if (two_hill){    
    
    if (first_entry){

      std::cout << "first entry, model->time = " << model->time << "\n";

      double max_value_two_hill = 0.0;
      double dt = 1e-5; 

      double t_shifted_temp;

      // two hill scalars 
      double g1_temp, g2_temp;
      double two_hill_val;

      for (double t_temp=0; t_temp<model->cardiac_cycle_period; t_temp += dt){
        t_shifted_temp = t_temp - t_shift;
        g1_temp = (t_shifted_temp > 0) ? pow(t_shifted_temp/tau_1, m1) : 0.0;
        g2_temp = (t_shifted_temp > 0) ? pow(t_shifted_temp/tau_2, m2) : 0.0;
        two_hill_val = (g1_temp/(1.0 + g1_temp)) * (1.0/(1.0 + g2_temp));

        max_value_two_hill = std::max(max_value_two_hill, two_hill_val);
      }

      std::cout << "max_value_two_hill = " << max_value_two_hill << "\n";
      normalization_twohill = 1.0/max_value_two_hill;
      normalization_initialized = true; 
      first_entry = false;
    }
  
      if (!normalization_initialized){
        throw std::runtime_error("Normalization not initialized");
      }
  
      double t_shifted = t_in_cycle - t_shift;
  
      // two hill scalars 
      double g1 = (t_shifted > 0) ? pow(t_shifted/tau_1, m1) : 0.0;
      double g2 = (t_shifted > 0) ? pow(t_shifted/tau_2, m2) : 0.0;
  
      act_two_hill = normalization_twohill * (g1/(1.0 + g1)) * (1.0/(1.0 + g2));
  }
  else{
    // cos default 
    double t_contract = 0;
    if (t_in_cycle >= t_active) {
      t_contract = t_in_cycle - t_active;
    }

    double act = 0;
    if (t_contract <= t_twitch) {
      act = -0.5 * cos(2 * M_PI * t_contract / t_twitch) + 0.5;
    }
  }

  std::cout << "model->time = " << model->time << "\tact = " << act << "\tact_two_hill = " << act_two_hill << "\n";

  Vrest = (1.0 - act) * (Vrd - Vrs) + Vrs;
  Elas = (Emax - Emin) * act + Emin;
}
