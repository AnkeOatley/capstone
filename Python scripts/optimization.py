import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import simulationfunctions as simfunc

# SEBIR QSSF error objective function
def SEBIR_QSSF_objective_func(droop_bounds, inertia_value, P_value, T_value, SC_connected, app, event_time, BESS_freq_model, BESS_PQ_model, BESS_results_path, freq_data_name, base_ssfreq, SC_model, Rms, oInit, elmres):
    droop_value = droop_bounds[0] # Set current droop value
 
    # Run the BESS case
    dfs_BESS = simfunc.SEBIR_case_sim(droop_value, inertia_value, P_value, T_value, SC_connected, app, event_time, BESS_freq_model, BESS_PQ_model, BESS_results_path, freq_data_name, Rms, oInit, elmres)
    value_nadir, value_nadirtime, value_rocofapprox, value_ssfreq, value_overshoot, value_overshoottime = simfunc.getNadirROCOFs_csv(dfs_BESS, event_time, freq_data_name)
    
    # Calculate the QSSF error between the freqeuncy responses of both the cases
    ss_error = abs(base_ssfreq-value_ssfreq)
    return ss_error 

# SEBIR RMS error objective function (no SC)
def SEBIR_RMS_objective_func(param_bounds, droop_optimal, SC_connected, app, event_time, BESS_freq_model, BESS_PQ_model, BESS_results_path, freq_data_name, dfs_b, Rms, oInit, elmres):
    droop_value = droop_optimal # Set value to optimal value
    inertia_value, P_value, T_value = param_bounds # Set current I, P and T values
    
    # Run the BESS case
    dfs_BESS = simfunc.SEBIR_case_sim(droop_value, inertia_value, P_value, T_value, SC_connected, app, event_time, BESS_freq_model, BESS_PQ_model, BESS_results_path, freq_data_name, Rms, oInit, elmres)
    
    # Calculate the RMS error between the freqeuncy responses of both the cases
    rms = mean_squared_error(dfs_BESS[(freq_data_name,"m:fehz in Hz")], dfs_b[(freq_data_name,"m:fehz in Hz")], squared = False) # Frequency error
    print(param_bounds)
    print(rms) # Uncomment to see the value printed on each iteration
    return rms 

# SEBIR RMS error objective function (with SC)
def SEBIR_RMS_objective_func_SC(param_bounds, droop_optimal, SC_connected, app, event_time, BESS_freq_model, BESS_PQ_model, BESS_results_path, freq_data_name, dfs_b, Rms, oInit, elmres):
    droop_value = droop_optimal # Set value to optimal value
    P_value, T_value = param_bounds # Set current P and T values
    inertia_value = 'N/A' # BESS inertia does not need to be changed 
    
    # Run the BESS case
    dfs_BESS = simfunc.SEBIR_case_sim(droop_value, inertia_value, P_value, T_value, SC_connected, app, event_time, BESS_freq_model, BESS_PQ_model, BESS_results_path, freq_data_name, Rms, oInit, elmres)
    
    # Calculate the RMS error between the freqeuncy responses of both the cases
    rms = mean_squared_error(dfs_BESS[(freq_data_name,"m:fehz in Hz")], dfs_b[(freq_data_name,"m:fehz in Hz")], squared = False) # Frequency error
    print(param_bounds)
    print(rms) # Uncomment to see the value printed on each iteration
    return rms 

# VSM QSSF error objective function
def VSM_QSSF_objective_func(droop_bounds, inertia_value, damping_value,SC_connected, app, event_time, BESS_freq_model, BESS_results_path, freq_data_name, base_ssfreq, Rms, oInit, elmres):
    droop_value = droop_bounds[0] # Set current droop value
    
    # Run the BESS case
    dfs_BESS = simfunc.VSM_case_sim(droop_value, inertia_value, damping_value, SC_connected, app, event_time, BESS_freq_model,  BESS_results_path, freq_data_name, Rms, oInit, elmres)
    value_nadir, value_nadirtime, value_rocofapprox, value_ssfreq, value_overshoot, value_overshoottime = simfunc.getNadirROCOFs_csv(dfs_BESS, event_time, freq_data_name)
    
    # Calculate the QSSF error between the freqeuncy responses of both the cases
    ss_error = abs(base_ssfreq-value_ssfreq)
    
    return ss_error 

# VSM RMS error objective function
def VSM_RMS_objective_func(param_bounds, droop_optimal, SC_connected, app, event_time, BESS_freq_model, BESS_results_path, freq_data_name, dfs_b, Rms, oInit, elmres):
    droop_value = droop_optimal # Set value to optimal value
    inertia_value = param_bounds[0] # Set to current ROCOF value
    damping_value = param_bounds[1] # Set current damping value

    # Run the BESS case
    dfs_BESS = simfunc.VSM_case_sim(droop_value, inertia_value, damping_value, SC_connected, app, event_time, BESS_freq_model,  BESS_results_path, freq_data_name, Rms, oInit, elmres)
    
    # Calculate the RMS error between the freqeuncy responses of both the cases
    rms = mean_squared_error(dfs_BESS[(freq_data_name,"m:fehz in Hz")], dfs_b[(freq_data_name,"m:fehz in Hz")], squared = False) # Frequency error
    print(param_bounds) 
    print(rms) # Uncomment to see the value printed on each iteration
    return rms 

# VSM RMS error objective function
def VSM_RMS_objective_func_SC(param_bounds, droop_optimal, SC_connected, app, event_time, BESS_freq_model, BESS_results_path, freq_data_name, dfs_b, Rms, oInit, elmres):
    droop_value = droop_optimal # Set value to optimal value
    inertia_value = 0 # Set to 0 as SC is enabled
    damping_value = param_bounds[0] # Set current damping value

    # Run the BESS case
    dfs_BESS = simfunc.VSM_case_sim(droop_value, inertia_value, damping_value, SC_connected, app, event_time, BESS_freq_model,  BESS_results_path, freq_data_name, Rms, oInit, elmres)
    
    # Calculate the RMS error between the freqeuncy responses of both the cases
    rms = mean_squared_error(dfs_BESS[(freq_data_name,"m:fehz in Hz")], dfs_b[(freq_data_name,"m:fehz in Hz")], squared = False) # Frequency error
    print(param_bounds) 
    print(rms) # Uncomment to see the value printed on each iteration
    return rms 