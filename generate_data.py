import pandas as pd
import numpy as np
import datetime
from datetime import timedelta

# --- CONFIGURATION ---
NUM_AREAS = 10
LOCS_PER_AREA = 10
START_DATE = datetime.datetime(2025, 1, 1)
DAYS_TO_SIMULATE = 5 
PACKAGES_PER_DAY = 5000 
FAILURE_RATE = 0.15   

# Failure Modes (Used to inject anomalies)
ROOT_CAUSE_PROBS = {
    'LATE_ARRIVAL': 0.4,   
    'MISLOAD': 0.2,        
    'YARD_DELAY': 0.3,     
    'CAPACITY_FAIL': 0.1   
}

np.random.seed(42)

def generate_network():
    locations = []
    loc_registry = {}
    
    for area_i in range(NUM_AREAS):
        for loc_i in range(LOCS_PER_AREA):
            loc_id = f"A{area_i}_L{loc_i}"
            is_hub = (loc_i == 0)
            busyness = np.random.uniform(0.8, 1.2) 
            loc_data = {'loc_id': loc_id, 'area': area_i, 'is_hub': is_hub, 'busyness': busyness}
            locations.append(loc_data)
            loc_registry[loc_id] = loc_data
            
    return pd.DataFrame(locations), loc_registry

def generate_sort_calendar(loc_registry, sim_dates):
    calendar = {}
    for date in sim_dates:
        date_str = date.strftime('%Y-%m-%d')
        for loc_id, data in loc_registry.items():
            start_hour = 20 if data['is_hub'] else 17
            duration = 6
            start_dt = date + timedelta(hours=start_hour)
            end_dt = start_dt + timedelta(hours=duration)
            cut_dt = end_dt - timedelta(minutes=45) 
            key = (loc_id, date_str)
            calendar[key] = {'start': start_dt, 'cut': cut_dt, 'end': end_dt}
    return calendar

def simulate_package_journey(pkg_id, origin, dest, pickup_date, calendar, loc_registry):
    scans = []
    path = [origin, f"A{loc_registry[origin]['area']}_L0"]
    if loc_registry[origin]['area'] != loc_registry[dest]['area']:
        path.append(f"A{loc_registry[dest]['area']}_L0")
    path.append(dest)
    
    will_fail = np.random.random() < FAILURE_RATE
    cause = np.random.choice(list(ROOT_CAUSE_PROBS.keys()), p=list(ROOT_CAUSE_PROBS.values())) if will_fail else "NONE"
    
    curr_time = pickup_date + timedelta(hours=14)
    fail_step_idx = np.random.randint(1, len(path)-1) if will_fail else -1

    for i, loc_id in enumerate(path):
        is_last_step = (i == len(path) - 1)
        loc_meta = loc_registry[loc_id]
        
        day_offset = 0 if i == 0 else (curr_time.date() - pickup_date.date()).days
        cal_date_str = (pickup_date + timedelta(days=day_offset)).strftime('%Y-%m-%d')
        sort_window = calendar.get((loc_id, cal_date_str), {
            'start': curr_time, 'cut': curr_time+timedelta(hours=4), 'end': curr_time+timedelta(hours=5)
        })
        
        sort_connection_time = (sort_window['start'] - curr_time).total_seconds() / 60.0
        time_to_cut = (sort_window['cut'] - curr_time).total_seconds() / 60.0
        facility_load = loc_meta['busyness'] * np.random.uniform(0.9, 1.3)
        if cause == 'CAPACITY_FAIL' and i == fail_step_idx: facility_load = 1.5 
            
        is_trigger_scan = False
        delay_minutes = np.random.randint(15, 45) 
        lane_id = f"LANE_{loc_id[-3:]}_{np.random.randint(1, 10)}"
        
        if will_fail and i == fail_step_idx:
            is_trigger_scan = True
            if cause == 'LATE_ARRIVAL':
                delay_added = max(0, time_to_cut) + np.random.randint(10, 60)
                curr_time += timedelta(minutes=delay_added)
                time_to_cut = (sort_window['cut'] - curr_time).total_seconds() / 60.0
            elif cause == 'YARD_DELAY': delay_minutes = 300 
            elif cause == 'MISLOAD': delay_minutes = 120; lane_id = "LANE_ERROR_999"
            
        scan_event = 'Delivery' if is_last_step else 'Sort'
        curr_time += timedelta(minutes=delay_minutes)
        
        scans.append({
            'PackageID': pkg_id, 'ScanTime': curr_time, 'LocationID': loc_id, 'Event': scan_event,
            'LaneID': lane_id, 
            'Sort_Connection_Mins': sort_connection_time, 'Time_To_Cut_Mins': time_to_cut, 'Facility_Load': facility_load,               
            'Is_Failure_Trigger': 1 if is_trigger_scan else 0,
            'Root_Cause_Type': cause if is_trigger_scan else 'NONE'
        })
        curr_time += timedelta(hours=np.random.randint(2, 5))

    return scans

def run_simulation():
    print("--- Generating Data (Supervised + Rich Physics) ---")
    loc_df, loc_registry = generate_network()
    dates = [START_DATE + timedelta(days=i) for i in range(DAYS_TO_SIMULATE + 2)]
    calendar = generate_sort_calendar(loc_registry, dates)
    
    all_scans = []
    loc_list = list(loc_registry.keys())
    
    for day_i in range(DAYS_TO_SIMULATE):
        curr_date = START_DATE + timedelta(days=day_i)
        vol = PACKAGES_PER_DAY
        origins = np.random.choice(loc_list, size=vol)
        dests = np.random.choice(loc_list, size=vol)
        
        for p_i in range(vol):
            if origins[p_i] == dests[p_i]: continue
            pkg_id = f"PKG_{day_i}_{p_i}"
            all_scans.extend(simulate_package_journey(pkg_id, origins[p_i], dests[p_i], curr_date, calendar, loc_registry))
            
    df_scans = pd.DataFrame(all_scans)
    df_packages = df_scans.groupby('PackageID').agg({
        'Is_Failure_Trigger': 'max' # Binary label for the whole package
    }).reset_index().rename(columns={'Is_Failure_Trigger': 'FailedService'})
    
    return df_packages, df_scans, None, None
