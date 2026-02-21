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

COMMIT_DAYS_SAME_AREA = 1
COMMIT_DAYS_CROSS_AREA = 2

# Failure Modes (Used to inject anomalies)
ROOT_CAUSE_PROBS = {
    'LATE_ARRIVAL': 0.4,
    'MISLOAD': 0.2,
    'YARD_DELAY': 0.3,
    'CAPACITY_FAIL': 0.1
}

# Scan message types per facility role
ORIGIN_SCANS = ['trailer_load']
INTERMEDIATE_REQUIRED = ['sorter_scan', 'trailer_load_scan']
INTERMEDIATE_OPTIONAL = ['arrival_scan', 'unload_scan', 'quality_scan']
DESTINATION_SCANS = ['van_scan', 'delivered']

# Canonical ordering for intermediate scans
_INTERMEDIATE_ORDER = ['arrival_scan', 'unload_scan', 'sorter_scan', 'quality_scan', 'trailer_load_scan']

np.random.seed(42)

def generate_network():
    locations = []
    loc_registry = {}

    for area_i in range(NUM_AREAS):
        for loc_i in range(LOCS_PER_AREA):
            loc_id = f"A{area_i}_L{loc_i}"
            is_hub = (loc_i == 0)
            loc_data = {'loc_id': loc_id, 'area': area_i, 'is_hub': is_hub}
            locations.append(loc_data)
            loc_registry[loc_id] = loc_data

    return pd.DataFrame(locations), loc_registry

def _get_facility_scans(facility_type):
    if facility_type == 'origin':
        return list(ORIGIN_SCANS)
    elif facility_type == 'destination':
        return list(DESTINATION_SCANS)
    else:
        num_optional = np.random.randint(0, len(INTERMEDIATE_OPTIONAL) + 1)
        if num_optional > 0:
            chosen = set(np.random.choice(INTERMEDIATE_OPTIONAL, size=num_optional, replace=False))
        else:
            chosen = set()
        all_scans = set(INTERMEDIATE_REQUIRED) | chosen
        return [s for s in _INTERMEDIATE_ORDER if s in all_scans]

def simulate_package_journey(pkg_id, origin, dest, pickup_date, loc_registry):
    scans = []

    # Build path: origin -> origin_hub -> dest_hub (if cross-area) -> destination
    path = [origin, f"A{loc_registry[origin]['area']}_L0"]
    is_cross_area = loc_registry[origin]['area'] != loc_registry[dest]['area']
    if is_cross_area:
        path.append(f"A{loc_registry[dest]['area']}_L0")
    path.append(dest)
    # Remove consecutive duplicates (e.g., origin IS the hub)
    path = [path[0]] + [path[i] for i in range(1, len(path)) if path[i] != path[i-1]]

    # Commit date
    transit_days = COMMIT_DAYS_CROSS_AREA if is_cross_area else COMMIT_DAYS_SAME_AREA
    commit_date = (pickup_date + timedelta(days=transit_days)).date()

    # Failure setup
    will_fail = np.random.random() < FAILURE_RATE
    cause = np.random.choice(
        list(ROOT_CAUSE_PROBS.keys()), p=list(ROOT_CAUSE_PROBS.values())
    ) if will_fail else "NONE"

    # Failure happens at an intermediate facility (not origin or destination)
    intermediate_indices = list(range(1, len(path) - 1))
    if will_fail and len(intermediate_indices) > 0:
        fail_facility_idx = int(np.random.choice(intermediate_indices))
    else:
        will_fail = False
        cause = "NONE"
        fail_facility_idx = -1

    curr_time = pickup_date + timedelta(hours=14)  # pickup at 2PM

    for i, loc_id in enumerate(path):
        is_origin = (i == 0)
        is_destination = (i == len(path) - 1)
        is_trigger_facility = will_fail and i == fail_facility_idx

        if is_origin:
            facility_type = 'origin'
        elif is_destination:
            facility_type = 'destination'
        else:
            facility_type = 'intermediate'

        facility_scans = _get_facility_scans(facility_type)

        # Inter-facility transit (not for origin)
        if not is_origin:
            transit_hours = np.random.randint(3, 9)
            if is_trigger_facility and cause == 'LATE_ARRIVAL':
                transit_hours += np.random.randint(30, 61)
            curr_time += timedelta(hours=transit_hours)

        for scan_idx, scan_event in enumerate(facility_scans):
            # Intra-facility delay between consecutive scans
            if scan_idx > 0:
                intra_delay = np.random.randint(15, 46)
                # YARD_DELAY / CAPACITY_FAIL: large delay between 1st and 2nd scan
                if is_trigger_facility and scan_idx == 1:
                    if cause == 'YARD_DELAY':
                        intra_delay += np.random.randint(600, 1200)
                    elif cause == 'CAPACITY_FAIL':
                        intra_delay += np.random.randint(480, 720)
                curr_time += timedelta(minutes=intra_delay)

            lane_id = f"LANE_{loc_id}_{np.random.randint(1, 10)}"

            # MISLOAD: wrong lane on trailer_load_scan + extra delay
            if is_trigger_facility and cause == 'MISLOAD' and scan_event == 'trailer_load_scan':
                lane_id = "LANE_ERROR_999"
                curr_time += timedelta(minutes=np.random.randint(180, 420))

            # Determine trigger scan for step-level labels
            is_trigger_scan = False
            if is_trigger_facility:
                if cause == 'LATE_ARRIVAL' and scan_idx == 0:
                    is_trigger_scan = True
                elif cause in ('YARD_DELAY', 'CAPACITY_FAIL') and scan_idx == 1:
                    is_trigger_scan = True
                elif cause == 'MISLOAD' and scan_event == 'trailer_load_scan':
                    is_trigger_scan = True

            scans.append({
                'PackageID': pkg_id,
                'ScanTime': curr_time,
                'LocationID': loc_id,
                'Event': scan_event,
                'LaneID': lane_id,
                'Commit_Date': commit_date,
                'Destination': dest,
                'Is_Failure_Trigger': 1 if is_trigger_scan else 0,
                'Root_Cause_Type': cause if is_trigger_scan else 'NONE'
            })

        # Inter-facility gap after processing (not after destination)
        if not is_destination:
            curr_time += timedelta(hours=np.random.uniform(0.5, 2.0))

    return scans

def run_simulation():
    print("--- Generating Data ---")
    loc_df, loc_registry = generate_network()

    all_scans = []
    loc_list = list(loc_registry.keys())

    for day_i in range(DAYS_TO_SIMULATE):
        curr_date = START_DATE + timedelta(days=day_i)
        origins = np.random.choice(loc_list, size=PACKAGES_PER_DAY)
        dests = np.random.choice(loc_list, size=PACKAGES_PER_DAY)

        for p_i in range(PACKAGES_PER_DAY):
            if origins[p_i] == dests[p_i]:
                continue
            pkg_id = f"PKG_{day_i}_{p_i}"
            all_scans.extend(simulate_package_journey(
                pkg_id, origins[p_i], dests[p_i], curr_date, loc_registry
            ))

    df_scans = pd.DataFrame(all_scans)

    # Derive FailedService: final scan date > commit date
    delivered = df_scans[df_scans['Event'] == 'delivered'].copy()
    delivered['Delivery_Date'] = pd.to_datetime(delivered['ScanTime']).dt.date
    delivered['FailedService'] = (delivered['Delivery_Date'] > delivered['Commit_Date']).astype(int)

    df_packages = delivered[['PackageID', 'FailedService', 'Commit_Date', 'Destination']].copy()

    # Build trigger_info for step-level labels
    triggers = df_scans[df_scans['Is_Failure_Trigger'] == 1][['PackageID', 'ScanTime']].copy()
    triggers = triggers.rename(columns={'ScanTime': 'Trigger_Time'})
    trigger_info = triggers.groupby('PackageID').first().reset_index()

    return df_packages, df_scans, trigger_info, None
