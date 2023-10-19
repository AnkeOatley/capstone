 # Import packages
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import plotly.express as px
import matplotlib.pyplot as plt

# Clear simulation event and data 
def clearSimEvents(app):
    if app.GetActiveStudyCase()==None:
        raise Exception('Function requires Active Study Case') # must have an active study case.
    
    faultFolder = app.GetFromStudyCase("Simulation Events/Fault.IntEvt") # get folder containing events like load events
    cont = faultFolder.GetContents()
    for obj in cont:
        obj.Delete() #delete events currently stored in this folder

# Get elements which are used by the calculations filtered by type of object
def getElements(app):
    elements = {}
    elements['buses'] = app.GetCalcRelevantObjects("*.ElmTerm")  #buses
    elements['syncgens'] = app.GetCalcRelevantObjects("*.ElmSym")    # Synchronous machines
    elements['loads'] = app.GetCalcRelevantObjects("*.ElmLod")   # Loads
    elements['signals'] = app.GetCalcRelevantObjects("*.BlkSig") # Signals
    return elements

# Add result variables for each element
def defineresultvars(app, elements):
    elmres = app.GetFromStudyCase('*.ElmRes')
    elmres.Clear()
    for bus in elements['buses']:
        elmres.AddVars(bus, "m:fe", #Electrical Frequency (p.u)
                            "m:dfehz", #Deviation of Electrical Frequency (HZ)
                            "m:dfedt", # Derivative of Electrical Frequency (p.u./s)
                            "m:fehz", #Eletrical Frequency (Hz)
                            "m:u" #Voltage Magnitude (p.u)
        )
    for gen in elements['syncgens']:
        elmres.AddVars(gen, "m:P:bus1", #Active Power (MW)
                            "m:Q:bus1", #Reactive Power (MW)
                            "m:u:bus1", #Voltage Magnitude (p.u)
                            "m:i:bus1" #Current Magnitude (p.u)
        )
    for load in elements['loads']:
        elmres.AddVars(load, "m:P:bus1", #Active Power (MW)
                             "m:Q:bus1", #Reactive Power (MW)
                             "m:u:bus1", #Voltage Magnitude (p.u)
                             "m:i:bus1" #Current Magnitude (p.u)
        )
    for sig in elements['signals']:
        elmres.AddVars(sig, "s:SOC") # SOC
    return elmres

# Define a load step change event
def defineLoadEvent(app, outage_model_name, event_time, load_change):
    EvtFold = app.GetFromStudyCase('IntEvt')
    LoadEvent = EvtFold.CreateObject('EvtLod','Load_Event')
    LoadEvent.p_target = app.GetCalcRelevantObjects(outage_model_name)[0] # select load
    LoadEvent.time = event_time  
    LoadEvent.iopt_type = 0 # set change type to step (ramp=1)
    LoadEvent.dP = load_change # set the percentage change

# Define a load step change event
def defineRampEvent(app, outage_model_name, event_time, load_change):
    Load_model = app.GetCalcRelevantObjects(outage_model_name)[0]
    Load_model.ramp_type = 1 # allow ramps
    EvtFold = app.GetFromStudyCase('IntEvt')
    LoadEvent = EvtFold.CreateObject('EvtLod','Load_Event')
    LoadEvent.p_target = app.GetCalcRelevantObjects(outage_model_name)[0] # select load
    LoadEvent.time = event_time  
    LoadEvent.iopt_type = 1 # set change type to step (ramp=1)
    LoadEvent.t_ramp = 60 # 60 second ramp time
    LoadEvent.dP = load_change # set the percentage change

# Define a generator disconection event
def defineGenEvent(app, outage_model_name, event_time, i):
    EvtFold = app.GetFromStudyCase('IntEvt')
    GenEvent = EvtFold.CreateObject('EvtSwitch','Switch Event')
    GenEvent.p_target = app.GetCalcRelevantObjects(outage_model_name)[i] # select generator
    GenEvent.time = event_time  
    GenEvent.i_switch = 0 #Open
    GenEvent.i_allph 

# Set up the simulation inital conditions
def setupsimulation(app, start_time, stop_time):
    oInit = app.GetFromStudyCase('ComInc') # get initial condition calculation object
    Rms = app.GetFromStudyCase('ComSim') # get RMS-simulation object
    oInit.tstart = start_time # simulation start (ms)
    Rms.tstop = stop_time  # simulation end (sec)
    #Rms.dtgrd = 0.01   # step size (sec)
    return Rms, oInit

# Gathers results for each variable of interest in csv file
def getResultscsv(file_path, app, elmres):
    # Get results file, load data and find number or rows
    elmres.Load()
    # Export results from powerfactory
    comres = app.GetFromStudyCase('ComRes')
    comres.pResult = elmres
    comres.iopt_exp = 6
    comres.f_name = file_path
    comres.iopt_csel = 0
    comres.iopt_tsel = 0
    comres.iopt_locn = 1
    comres.ciopt_head = 1
    comres.Execute()
    # Read the csv file
    df = pd.read_csv(file_path, header=[0,1])
    return df

# Returns nadir value, ROCOF and steady state frequency
def getNadirROCOFs_csv(df, loadeventtime,freq_data_name):
    interval = 0.5
    value_nadir = df[(freq_data_name,"m:fehz in Hz")].min()
    nadir_index = df[(freq_data_name,"m:fehz in Hz")].idxmin() 
    overshoot_array = df[(freq_data_name,"m:fehz in Hz")][nadir_index:]
    value_overshoot = overshoot_array.max()
    overshoot_index = overshoot_array.idxmax() 
    value_nadirtime = df[("All calculations", "b:tnow in s" )].iloc[nadir_index]
    value_overshoottime = df[("All calculations", "b:tnow in s" )].iloc[overshoot_index]
    idx_start = df[df[("All calculations", "b:tnow in s" )] >=loadeventtime].index[0] 
    freq_start = df[(freq_data_name,"m:fehz in Hz")][idx_start]
    idx_later = df[df[("All calculations", "b:tnow in s" )] >=loadeventtime+interval].index[0] 
    freq_later = df[(freq_data_name,"m:fehz in Hz")][idx_later]
    value_rocofapprox = (freq_later - freq_start)/interval
    value_ssfreq = df[(freq_data_name,"m:fehz in Hz")].iloc[-1] # steady state value (ensuring sim time is sufficiently long)
    return value_nadir, value_nadirtime, value_rocofapprox, value_ssfreq, value_overshoot, value_overshoottime

# Set up base case for simulations
def base_case_setup(app, event, event_time, load_change, load_model_name, gen_model_name, sim_start_time, sim_stop_time, proj_name_base):
    # Activate base case project and study case
    project = app.ActivateProject(proj_name_base)
    if project == 1: # test to ensure project was found
        raise Exception("Could not activate project") 
    proj = app.GetActiveProject()
    oFolder_studycase = app.GetProjectFolder('study')
    oCase = oFolder_studycase.GetContents()[0]
    oCase.Activate()

    # Clear previous sim
    app.ResetCalculation()
    clearSimEvents(app)
    # Disable break button in main toobar (good to do before executing lots of python code)
    app.SetUserBreakEnabled(0)

    ## Set up access to elements in system 
    elements = getElements(app)
    #print(elements)

    ## Define result variables
    elmres = defineresultvars(app, elements) # set which result varaibles to be tracked

    ## Simulation event definition 
    if event == 0: # Load step
        defineLoadEvent(app, load_model_name, event_time, load_change)

    if event == 1:  # Load ramp
        defineRampEvent(app, load_model_name, event_time, load_change)

    # Set start and stop time and define intial condition and simulation objects
    Rms, oInit = setupsimulation(app, sim_start_time, sim_stop_time)

    return Rms, oInit, elmres

# Function for toggling on or off a SC
def toggle_SC(app, SC_model, SC_connected):
    SC_element = app.GetCalcRelevantObjects(SC_model)[0]
    if(SC_connected):
        SC_element.outserv = 0
        print(str(SC_model) + ' enabled')
    else:
        SC_element.outserv = 1
        print(str(SC_model) + ' disabled')
    return

# Function for toggling on or off AGC
def toggle_AGC(app, AGC_model, AGC_connected):
    SC_element = app.GetCalcRelevantObjects(AGC_model)[0]
    if(AGC_connected):
        SC_element.outserv = 0
        print(str(AGC_model) + ' enabled')
    else:
        SC_element.outserv = 1
        print(str(AGC_model) + ' disabled')
    return

# Function for adjusting AGC gain in BESS
def set_AGC_gain(app, BESS_P_model, AGC_gain):
    P_element = app.GetCalcRelevantObjects(BESS_P_model)[0]
    P_element.C = AGC_gain
     
# Run the base case 
def base_case_sim(app, event_time, base_results_path, freq_data_name, Rms, oInit, elmres):
    # Set up objects to collect results from file
    dfs_b = []
    nadir_b = []
    nadir_time_b = []
    rocofapprox_b = []
    ssfreq_b = []
    overshoot_b = []
    overshoot_time_b = []

    # Run simulation
    oInit.Execute() #calculate initial conditions
    Rms.Execute()  # run simulation
    
    # Results collection
    df = getResultscsv(base_results_path, app, elmres)
    df["sim"] = "Base case" # Label results
    dfs_b.append(df)
    
    #Find nadir, rocof and steady state frequency value reached
    value_nadir, value_nadirtime, value_rocofapprox, value_ssfreq, value_overshoot, overshoot_time = getNadirROCOFs_csv(df, event_time, freq_data_name)
    nadir_b.append(value_nadir)
    nadir_time_b.append(value_nadirtime)
    rocofapprox_b.append(value_rocofapprox)
    ssfreq_b.append(value_ssfreq) 
    overshoot_b.append(value_overshoot)
    overshoot_time_b.append(overshoot_time)
    dfs_b = pd.concat(dfs_b)

    # Check the voltages
    for col_name in dfs_b.columns[1:]:
        if "m:u in p.u." in col_name:
            # Print a warning if over or under voltage 
            if max(dfs_b[col_name])>1.06:
                print("OVERVOLTAGE WARNING: " + col_name[0])
            # Print a warning if over or under voltage 
            if min(dfs_b[col_name])<0.94:
                print("UNDERVOLTAGE WARNING: " + col_name[0])

    return dfs_b

# Set up SEBIR case for simualtions
def SEBIR_case_setup(app, event, event_time, load_change, SC_connected, load_model_name, gen_model_name, sim_start_time, sim_stop_time, proj_name_BESS, BESS_series_cells, BESS_par_cells, BESS_DC_voltage, BESS_soc0, BESS_rated_power, BESS_setpoint, BESS_battery_model, BESS_PWM_model, BESS_DC_bus_model, BESS_DC_source_model, BESS_freq_model, SC_model):
    # Activate BESS case project and study case
    project = app.ActivateProject(proj_name_BESS)
    if project == 1: #test to ensure project was found
        raise Exception("Could not activate project") 
    proj = app.GetActiveProject()
    oFolder_studycase = app.GetProjectFolder('study')
    oCase = oFolder_studycase.GetContents()[0]
    oCase.Activate()

    # Clear previous sim
    app.ResetCalculation()
    clearSimEvents(app)
    # Disable break button in main toobar (good to do before executing lots of python code)
    app.SetUserBreakEnabled(0)

    # Set up access to elements in system 
    elements = getElements(app)

    # Define result variables
    elmres = defineresultvars(app, elements) # set which result varaibles to be tracked

    # Simulation event definition 
    if event == 0: # Load step
        defineLoadEvent(app, load_model_name, event_time, load_change)
    if event == 1:  # Load ramp
        defineRampEvent(app, load_model_name, event_time, load_change)

    # Set start and stop time and define intial condition and simulation objects
    Rms, oInit = setupsimulation(app, sim_start_time, sim_stop_time)

    # Access parameters of interest
    battery_element = app.GetCalcRelevantObjects(BESS_battery_model)[0]
    pwm_element = app.GetCalcRelevantObjects(BESS_PWM_model)[0]
    freq_element = app.GetCalcRelevantObjects(BESS_freq_model)[0]
    DC_bus_element = app.GetCalcRelevantObjects(BESS_DC_bus_model)[0]
    DC_source_element = app.GetCalcRelevantObjects(BESS_DC_source_model)[0]
    
    # Set up battery pack parameters
    battery_element.nSerialCells = int(BESS_series_cells)
    battery_element.nParallelCells = int(BESS_par_cells)
    battery_element.Unom = BESS_DC_voltage/1000
    battery_element.SOC0 = BESS_soc0
    pwm_element.psetp = BESS_setpoint
    pwm_element.Snom = BESS_rated_power
    pwm_element.nparnum = 1
    DC_bus_element.unknom = BESS_DC_voltage/1000
    DC_source_element.Unom = BESS_DC_voltage/1000

    # Set up inertial response
    if(SC_connected):
        toggle_SC(app, SC_model, True) # Switch on SC
        freq_element.frq_ctrl = 0 # If SC is enabled, disable the inertial response of the BESS
        pwm_element.i_acdc =  5 # SC will control the voltage, so set BESS to PQ mode
    else:
        toggle_SC(app, SC_model, False) # Switch off SC
        freq_element.frq_ctrl = 2 # If SC is disabled, enable the inertial response of the BESS
        pwm_element.i_acdc = 4 # Set BESS to voltage control mode
    return Rms, oInit, elmres

# Run SEBIR case
def SEBIR_case_sim(droop_value, inertia_value, P_value, T_value, SC_connected, app, event_time, BESS_freq_model, BESS_PQ_model, BESS_results_path, freq_data_name, Rms, oInit, elmres):
    # Access battery model which has the parameter of interest
    freq_element = app.GetCalcRelevantObjects(BESS_freq_model)[0]
    pq_element = app.GetCalcRelevantObjects(BESS_PQ_model)[0]
    
    # Set up objects to collect results from file
    dfs_BESS = []
    nadir_BESS = []
    nadir_time_BESS = []
    rocofapprox_BESS = []
    ssfreq_BESS = []
    overshoot_BESS = []
    overshoot_time_BESS = []

    if not SC_connected:
        freq_element.K_rocof =  inertia_value # Inertia constant is equal to 1/2H for this controller
    freq_element.droop = droop_value
    pq_element.Kp = P_value
    pq_element.Tip = T_value
    
    # Run simulation and collect results
    oInit.Execute() #calculate initial conditions
    Rms.Execute()  # run simulation
    df = getResultscsv(BESS_results_path, app, elmres)
    if SC_connected:
        df["sim"] = "BESS case with SC"
    else:
        df["sim"] = "BESS case"
    dfs_BESS.append(df)
    value_nadir, value_nadirtime, value_rocofapprox, value_ssfreq, value_overshoot, overshoot_index = getNadirROCOFs_csv(df, event_time, freq_data_name)
    nadir_BESS.append(value_nadir)
    nadir_time_BESS.append(value_nadirtime)
    rocofapprox_BESS.append(value_rocofapprox)
    ssfreq_BESS.append(value_ssfreq)
    overshoot_BESS.append(value_overshoot)
    overshoot_time_BESS.append(overshoot_index)
    dfs_BESS = pd.concat(dfs_BESS)
    
    # Print a warning if over or under voltage 
    for col_name in df.columns[1:]:
        if "m:u in p.u." in col_name:
            if df[col_name].max() > 1.06:
                print("OVERVOLTAGE WARNING: "  + col_name[0])
    
            if df[col_name].min() < 0.94:
                print("UNDERVOLTAGE WARNING: "  + col_name[0])

    return dfs_BESS

# Set up VSM case for simualtions
def VSM_case_setup(app, event, event_time, load_change, lv_value, q_value, SC_connected, load_model_name, gen_model_name, sim_start_time, sim_stop_time, proj_name_BESS, BESS_series_cells, BESS_par_cells, BESS_DC_voltage, BESS_soc0, BESS_rated_power, BESS_setpoint, BESS_battery_model, BESS_PWM_model, BESS_impedance_model, BESS_RPC_model, BESS_CC_model, BESS_freq_model, SC_model):
    # Activate BESS case project and study case
    project = app.ActivateProject(proj_name_BESS)
    if project == 1: #test to ensure project was found
        raise Exception("Could not activate project") 
    proj = app.GetActiveProject()
    oFolder_studycase = app.GetProjectFolder('study')
    oCase = oFolder_studycase.GetContents()[0]
    oCase.Activate()

    # Clear previous sim
    app.ResetCalculation()
    clearSimEvents(app)
    # Disable break button in main toobar (good to do before executing lots of python code)
    app.SetUserBreakEnabled(0)

    # Set up access to elements in system 
    elements = getElements(app)

    # Define result variables
    elmres = defineresultvars(app, elements) # set which result varaibles to be tracked

    # Simulation event definition 
    if event == 0: # Load step
        defineLoadEvent(app, load_model_name, event_time, load_change)
    if event == 1:  # Load ramp
        defineRampEvent(app, load_model_name, event_time, load_change) #defineGenEvent(app, gen_model_name, event_time, 1)
        

    # Set start and stop time and define intial condition and simulation objects
    Rms, oInit = setupsimulation(app, sim_start_time, sim_stop_time)

    # Access parameters of interest
    battery_element = app.GetCalcRelevantObjects(BESS_battery_model)[0]
    pwm_element = app.GetCalcRelevantObjects(BESS_PWM_model)[0]
    freq_element = app.GetCalcRelevantObjects(BESS_freq_model)[0]
    cc_element = app.GetCalcRelevantObjects(BESS_CC_model)[0]
    impedance_element = app.GetCalcRelevantObjects(BESS_impedance_model)[0]
    reactive_element = app.GetCalcRelevantObjects(BESS_RPC_model)[0]
    
    # Set up battery pack parameters
    battery_element.nSerialCells = int(BESS_series_cells)
    battery_element.nParallelCells = int(BESS_par_cells)
    battery_element.Unom = BESS_DC_voltage/1000
    battery_element.SOC0 = BESS_soc0
    pwm_element.pgini = BESS_setpoint
    pwm_element.sgn = BESS_rated_power
    pwm_element.nparnum = 1
    cc_element.Udc_base = BESS_DC_voltage

    # Virtual inductance 
    freq_element.l = lv_value # Tune to make the VSM stable
    impedance_element.lv = lv_value # Tune to make the VSM stable

    # Reactive droop
    reactive_element.kq = q_value # Reactive droop control value

     # Set up inertial response
    if(SC_connected):
        toggle_SC(app, SC_model, True) # Switch on SC
        freq_element.Ta =  0 # If SC is enabled, disable the inertial response of the BESS
        pwm_element.av_mode = 'constq' # SC will control the voltage, so set BESS to PQ mode
    else:
        toggle_SC(app, SC_model, False) # Switch off SC
        pwm_element.av_mode = 'constv' # Set BESS to voltage control mode

    return Rms, oInit, elmres

# Run VSM case
def VSM_case_sim(droop_value, inertia_value, damping_value, SC_connected, app, event_time, BESS_freq_model,  BESS_results_path, freq_data_name, Rms, oInit, elmres):
    # Access battery model which has the parameter of interest
    freq_element = app.GetCalcRelevantObjects(BESS_freq_model)[0]

    # Set up objects to collect results from file
    dfs_BESS = []
    nadir_BESS = []
    nadir_time_BESS = []
    rocofapprox_BESS = []
    ssfreq_BESS = []
    overshoot_BESS = []
    overshoot_time_BESS = []

    # Set param values 
    freq_element.kw = droop_value
    freq_element.Ta = inertia_value
    freq_element.kd = damping_value
    
    # Run simulation and collect results
    oInit.Execute() #calculate initial conditions
    Rms.Execute()  # run simulation
    df = getResultscsv(BESS_results_path, app, elmres)
    if SC_connected:
        df["sim"] = "BESS case with SC"
    else:
        df["sim"] = "BESS case"
    dfs_BESS.append(df)
    value_nadir, value_nadirtime, value_rocofapprox, value_ssfreq, value_overshoot, overshoot_index = getNadirROCOFs_csv(df, event_time, freq_data_name)
    nadir_BESS.append(value_nadir)
    nadir_time_BESS.append(value_nadirtime)
    rocofapprox_BESS.append(value_rocofapprox)
    ssfreq_BESS.append(value_ssfreq)
    overshoot_BESS.append(value_overshoot)
    overshoot_time_BESS.append(overshoot_index)
    dfs_BESS = pd.concat(dfs_BESS)
    
    # Print a warning if over or under voltage 
    for col_name in df.columns[1:]:
        if "m:u in p.u." in col_name:
            if df[col_name].max() > 1.06:
                print("OVERVOLTAGE WARNING: "  + col_name[0])
    
            if df[col_name].min() < 0.94:
                print("UNDERVOLTAGE WARNING: "  + col_name[0])

    return dfs_BESS


def plot_power_px(dfs_b, dfs_BESS, gen_data_name, BESS_data_name, BESS_power_name, SC_data_name, SC_connected):
    # Plot power outputs of base case generator and optimized BESS. If SC connected plot the combined response. 
    if(SC_connected):
        fig_Gen_P = px.line(
        x=dfs_BESS[("All calculations", "b:tnow in s")],
        y=dfs_BESS[(BESS_data_name, BESS_power_name)]+ dfs_BESS[(SC_data_name, 'm:Psum:bus1 in MW')],
        color=dfs_BESS[("sim", "")],
        )
    else:
        fig_Gen_P = px.line(
        x=dfs_BESS[("All calculations", "b:tnow in s")],
        y=dfs_BESS[(BESS_data_name, BESS_power_name)],
        color=dfs_BESS[("sim", "")],
        )
    new_trace = px.line(
        x=dfs_b[("All calculations", "b:tnow in s")],
        y=dfs_b[(gen_data_name, "m:P:bus1 in MW")],
        color=dfs_b[("sim", "")],
    )
    new_trace.data[0].update(opacity=0.5)  # Add the base case plot to the existing figure
    new_trace.update_traces(line=dict(dash='dash', color='red'))  # Set line style to dashed
    fig_Gen_P.add_trace(new_trace['data'][0])
    fig_Gen_P.update_layout(
        title='Power outputs',
        xaxis_title='Time (s)',
        yaxis_title='Active Power (MW)'
    )
    fig_Gen_P.show()

    return fig_Gen_P

def plot_freq_px(dfs_b, dfs_BESS, freq_data_name):
    # Plot frequency responses of base case generator and optimized BESS
    fig_Freq = px.line(
        x=dfs_BESS[("All calculations", "b:tnow in s")],
        y=dfs_BESS[(freq_data_name, "m:fehz in Hz")],
        color=dfs_BESS[("sim", "")],
    )
    new_trace = px.line(
        x=dfs_b[("All calculations", "b:tnow in s")],
        y=dfs_b[(freq_data_name, "m:fehz in Hz")],
        color=dfs_b[("sim", "")],
    )
    new_trace.data[0].update(opacity=0.5)  # Add the base case plot to the existing figure
    new_trace.update_traces(line=dict(dash='dash', color='red'))  # Set line style to dashed
    fig_Freq.add_trace(new_trace['data'][0])
    fig_Freq.update_layout(
        title='Frequency responses',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)'
    )
    fig_Freq.show()

    return fig_Freq

def plot_freq(dfs_b, dfs_BESS, freq_data_name, output_folder, data_length):
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    # Filter the data
    dfs_BESS_filtered = dfs_BESS[dfs_BESS[("All calculations", "b:tnow in s")] <= data_length]
    dfs_b_filtered = dfs_b[dfs_b[("All calculations", "b:tnow in s")] <= data_length]
 
    # Plot frequency responses of optimized BESS
    ax.plot(
        dfs_BESS_filtered[("All calculations", "b:tnow in s")],
        dfs_BESS_filtered[(freq_data_name, "m:fehz in Hz")],
        label='BESS Case: Frequency Response'
    )

    # Plot frequency responses of base case generator
    ax.plot(
        dfs_b_filtered[("All calculations", "b:tnow in s")],
        dfs_b_filtered[(freq_data_name, "m:fehz in Hz")],
        color='red',
        linestyle='--',
        label='Base Case: Frequency Response'
    )

    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

    # Add legend
    ax.legend()

    # Save the plot to the specified output folder
    plt.savefig(output_folder)

    # Show the plot
    plt.show()


def plot_power(dfs_b, dfs_BESS, gen_data_name, BESS_data_name, BESS_power_name, SC_data_name, SC_connected, output_folder, data_length):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Filter the data
    dfs_BESS_filtered = dfs_BESS[dfs_BESS[("All calculations", "b:tnow in s")] <= data_length]
    dfs_b_filtered = dfs_b[dfs_b[("All calculations", "b:tnow in s")] <= data_length]

    # Plot power outputs of base case generator and optimized BESS
    if SC_connected:
        ax.plot(
            dfs_BESS_filtered[("All calculations", "b:tnow in s")],
            dfs_BESS_filtered[(BESS_data_name, BESS_power_name)] + dfs_BESS_filtered[(SC_data_name, 'm:Psum:bus1 in MW')],
            label='BESS Case: Battery + SynCon Power',
        )
    else:
        ax.plot(
            dfs_BESS_filtered[("All calculations", "b:tnow in s")],
            dfs_BESS_filtered[(BESS_data_name, BESS_power_name)],
            label='BESS Case: Battery Power',
        )

    # Plot power output of the base case generator
    ax.plot(
        dfs_b_filtered[("All calculations", "b:tnow in s")],
        dfs_b_filtered[(gen_data_name, "m:P:bus1 in MW")],
        color='red',
        linestyle='--',
        label='Base Case: Generator Power'
    )

    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Active Power (MW)')

    # Add legend
    ax.legend()

    # Save the plot to the specified output folder
    plt.savefig(output_folder)

    # Show the plot
    plt.show()

def plot_power_reactive(dfs_b, dfs_BESS, gen_data_name, BESS_data_name, BESS_power_name, SC_data_name, SC_connected, output_folder, data_length):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Filter the data
    dfs_BESS_filtered = dfs_BESS[dfs_BESS[("All calculations", "b:tnow in s")] <= data_length]
    dfs_b_filtered = dfs_b[dfs_b[("All calculations", "b:tnow in s")] <= data_length]

    # Plot power outputs of base case generator and optimized BESS
    if SC_connected:
        ax.plot(
            dfs_BESS_filtered[("All calculations", "b:tnow in s")],
            dfs_BESS_filtered[(BESS_data_name, BESS_power_name)] + dfs_BESS_filtered[(SC_data_name, 'm:Qsum:bus1 in MW')],
            label='BESS Case: Battery + SynCon Power',
        )
    else:
        ax.plot(
            dfs_BESS_filtered[("All calculations", "b:tnow in s")],
            dfs_BESS_filtered[(BESS_data_name, BESS_power_name)],
            label='BESS Case: Battery Power',
        )

    # Plot power output of the base case generator
    ax.plot(
        dfs_b_filtered[("All calculations", "b:tnow in s")],
        dfs_b_filtered[(gen_data_name, "m:Q:bus1 in MW")],
        color='red',
        linestyle='--',
        label='Base Case: Generator Power'
    )

    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Reactive Power (MVAR)')

    # Add legend
    ax.legend()

    # Save the plot to the specified output folder
    plt.savefig(output_folder)

    # Show the plot
    plt.show()

def plot_voltage(dfs_b, dfs_BESS, volt_data_name, output_folder, data_length):
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    # Filter the data
    dfs_BESS_filtered = dfs_BESS[dfs_BESS[("All calculations", "b:tnow in s")] <= data_length]
    dfs_b_filtered = dfs_b[dfs_b[("All calculations", "b:tnow in s")] <= data_length]
 
    # Plot frequency responses of optimized BESS
    ax.plot(
        dfs_BESS_filtered[("All calculations", "b:tnow in s")],
        dfs_BESS_filtered[(volt_data_name, "m:u in p.u.")],
        label='BESS Case: Voltage Response'
    )

    # Plot frequency responses of base case generator
    ax.plot(
        dfs_b_filtered[("All calculations", "b:tnow in s")],
        dfs_b_filtered[(volt_data_name, "m:u in p.u.")],
        color='red',
        linestyle='--',
        label='Base Case: Voltage Response'
    )

    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (p.u.)')

    # Add legend
    ax.legend()

    # Save the plot to the specified output folder
    plt.savefig(output_folder)

    # Show the plot
    plt.show()

