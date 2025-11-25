# Digital Twin Dashboard - Data Requirements Analysis

## Overview
Based on the Digital Twin dashboard mockups, this document outlines the additional data columns needed to support a comprehensive Digital Twin visualization.

## Architecture Decision
**Important**: The Inference Log CSV contains only **prediction outputs** from the ML model. All **telematics/sensor data** should be stored in the **Historical Data CSV**. The Digital Twin screen will join these two datasets using **VIN** and **timestamp/date** to display comprehensive vehicle information.

### Data Flow:
1. **Historical Data CSV** → Contains all vehicle telematics data (sensor readings, DTCs, anomalies, etc.)
2. **Inference Log CSV** → Contains prediction results (pred_prob, pred_prob_pct, prescriptive_summary)
3. **Digital Twin View** → Joins both datasets on `VIN` + `timestamp`/`date` to show complete vehicle state

---

## Current Data Structure

### Historical Data CSV (Current Columns):
- `model`, `primary_failed_part`, `mileage_bucket`, `age_bucket`
- `date`, `claims_count`, `repairs_count`, `recalls_count`
- `vin`, `supplier_name`, `supplier_id`, `supplier_quality_score`, `defect_rate`
- `failure_description`, `current_lat`, `current_lon`, `city`
- `dealer_name`, `dealer_lat`, `dealer_lon`, `dealer_distance_km`
- `manufacturing_date`, `dtc_code`

### Inference Log CSV (Current Columns):
- `timestamp`, `vin`, `model`, `primary_failed_part`
- `mileage`, `age`, `pred_prob`, `pred_prob_pct`
- `lat`, `lon`, `prescriptive_summary`

**Note**: Inference Log should remain focused on prediction outputs only.

---

## Additional Data Requirements

### 1. Vehicle Overview Card

**Required Data:**
- ✅ `model` (already have in both)
- ✅ `vin` (already have in both - **join key**)
- ❌ `model_year` - Manufacturing year (e.g., 2022)
- ❌ `telematics_timestamp` - Timestamp when telematics data was received (for joining with inference log)
- ✅ `city` (already have in historical)
- ✅ `mileage` (already have in inference log - this is a **prediction input/feature**, not a sensor reading)
- ❌ `battery_soc` - Battery State of Charge percentage (0-100%) - **This is a sensor reading, NOT a prediction input**
- ❌ `vehicle_health_status` - Overall health status (HEALTHY, WARNING, CRITICAL) - **Can be derived from pred_prob_pct in inference log**

**Where to Add:**
- **Historical CSV**: `model_year`, `telematics_timestamp`, `battery_soc`, `city` (already have)
- **Inference Log CSV**: No changes needed (already has `timestamp` for joining, `mileage` for display)
  - **Note**: `mileage` and `age` in inference log are **prediction inputs/features** used by the ML model, not sensor readings
  - **Sensor readings** like `battery_soc` come from telematics and belong in Historical CSV only
- **Join Logic**: `historical.vin = inference_log.vin` AND `historical.telematics_timestamp ≈ inference_log.timestamp` (within time window, e.g., same day or ±1 hour)

---

### 2. Overall Health Score Gauge

**Required Data:**
- ❌ `overall_health_score` - Composite health score (0-100)
- ❌ `health_score_components` - Breakdown by subsystem (JSON or separate columns)

**Calculation Approach:**
- **Primary Source**: `pred_prob_pct` from Inference Log (main driver)
- **Secondary Factors**: Sensor readings from Historical CSV, DTC codes, component health
- Formula: `100 - (weighted_risk_score)` where risk_score combines pred_prob_pct + sensor anomalies + DTC severity

**Where to Add:**
- **Historical CSV**: Sensor readings and DTC data (inputs for calculation)
- **Inference Log CSV**: `pred_prob_pct` (already have - primary input)
- **Digital Twin View**: Calculate `overall_health_score` on-the-fly by joining both datasets
- **Alternative**: Store calculated `overall_health_score` in Historical CSV if calculated during data ingestion

---

### 3. Key Sensor Snapshot

**Required Data:**
- ❌ `battery_soc` - Battery State of Charge (%)
- ❌ `battery_voltage` - Battery voltage (volts, e.g., 355v)
- ❌ `ambient_temperature` - Ambient/vehicle temperature (°F or °C)
- ❌ `engine_rpm` - Engine RPM (for ICE/hybrid) or motor speed
- ❌ `coolant_temperature` - Coolant temperature (°F or °C)
- ❌ `vehicle_speed` - Current speed (mph or km/h)

**Where to Add:**
- **Historical CSV**: All sensor readings (real-time telematics data)
- **Inference Log CSV**: No changes needed
- **Join Logic**: Match by VIN + timestamp to get sensor readings for the same time as prediction

---

### 4. DTC Fault Panel

**Required Data:**
- ✅ `dtc_code` (already have in historical)
- ❌ `dtc_subsystem` - Subsystem affected (Battery, Engine, Brakes, Transmission, etc.)
- ❌ `dtc_severity` - Severity level (LOW, MEDIUM, HIGH, CRITICAL)
- ❌ `dtc_recommendation` - Action recommendation (e.g., "Replace hybrid battery pack")
- ❌ `dtc_explanation` - Detailed explanation of the fault
- ❌ `dtc_timestamp` - When DTC was first logged (can use `telematics_timestamp` or separate column)

**Where to Add:**
- **Historical CSV**: `dtc_code` (already have), `dtc_subsystem`, `dtc_severity`, `dtc_recommendation`, `dtc_explanation`, `dtc_timestamp`
- **Inference Log CSV**: No changes needed
- **Join Logic**: Match by VIN + timestamp to get active DTCs at time of prediction
- **Note**: DTC subsystem can be derived from DTC code using a lookup table (e.g., P0xxx = Battery, B1xxx = Body, C1xxx = Chassis)

---

### 5. Recent Anomalies Timeline

**Required Data:**
- ❌ `anomaly_type` - Type of anomaly (SOC drop, High temperature, DTC logged, Service event, etc.)
- ❌ `anomaly_timestamp` - When anomaly occurred
- ❌ `anomaly_severity` - Severity (INFO, WARNING, CRITICAL)
- ❌ `anomaly_description` - Human-readable description
- ❌ `anomaly_icon` - Icon type (lightning, thermometer, wrench, X, etc.)

**Where to Add:**
- **Historical CSV**: `anomaly_type`, `anomaly_timestamp`, `anomaly_severity`, `anomaly_description`, `anomaly_icon`
- **Inference Log CSV**: No changes needed
- **Join Logic**: Match by VIN to get all anomalies for the vehicle (can filter by timestamp range)
- **Alternative**: Can be calculated on-the-fly from sensor readings (e.g., detect SOC drop by comparing consecutive readings)

**Note:** Anomalies can be stored as separate rows in Historical CSV (one row per anomaly event) or as a JSON array column.

---

### 6. Component-Level Health (Battery, Engine, Brakes)

**Required Data:**

#### Battery Component:
- ❌ `battery_health_status` - Normal, Warning, Critical
- ❌ `battery_voltage_status` - Voltage reading status
- ❌ `battery_temperature` - Battery temperature (°C)
- ❌ `battery_charge_cycles` - Number of charge cycles
- ❌ `battery_degradation_pct` - Battery degradation percentage

#### Engine Component:
- ❌ `engine_health_status` - Normal, Warning, Critical
- ❌ `water_pump_speed` - Water pump RPM
- ❌ `coolant_temperature` - Coolant temp (°C)
- ❌ `oil_pressure` - Oil pressure (PSI)
- ❌ `engine_load` - Engine load percentage

#### Brakes Component:
- ❌ `brake_health_status` - Normal, Warning, Critical
- ❌ `brake_pressure` - Brake pressure (bar, e.g., 30 bar)
- ❌ `brake_pad_wear_pct` - Brake pad wear percentage (e.g., 24%)
- ❌ `brake_fluid_level` - Brake fluid level status
- ❌ `brake_torque` - Brake torque value

**Where to Add:**
- **Historical CSV**: All component-specific sensor readings and health statuses (real-time telematics data)
- **Inference Log CSV**: No changes needed
- **Join Logic**: Match by VIN + timestamp to get component health at time of prediction

**Component Health Status Calculation:**
- Can be derived from sensor readings and DTC codes in Historical CSV
- Normal: All sensors in range, no critical DTCs
- Warning: Some sensors out of range, minor DTCs
- Critical: Multiple sensors critical, major DTCs

---

## Summary: New Columns Required

### Historical Data CSV (Add these columns):
```
# Vehicle Info & Timestamp (for joining)
- model_year (integer) - Manufacturing year
- telematics_timestamp (datetime) - When telematics data was received (join key with inference_log.timestamp)

# Battery Data
- battery_soc (float, 0-100) - Battery State of Charge
- battery_voltage (float, volts) - Battery voltage (e.g., 355v)
- battery_temperature (float, °C) - Battery temperature
- battery_charge_cycles (integer) - Number of charge cycles
- battery_degradation_pct (float, 0-100) - Battery degradation
- battery_health_status (string) - Normal, Warning, Critical

# Engine Data
- engine_rpm (integer) - Engine RPM or motor speed
- water_pump_speed (integer, RPM) - Water pump speed
- coolant_temperature (float, °C) - Coolant temperature
- oil_pressure (float, PSI) - Oil pressure
- engine_load (float, 0-100%) - Engine load percentage
- engine_health_status (string) - Normal, Warning, Critical

# Brake Data
- brake_pressure (float, bar) - Brake pressure (e.g., 30 bar)
- brake_pad_wear_pct (float, 0-100%) - Brake pad wear (e.g., 24%)
- brake_fluid_level (string) - Brake fluid level status
- brake_torque (float) - Brake torque value
- brake_health_status (string) - Normal, Warning, Critical

# Environmental & Vehicle State
- ambient_temperature (float, °F or °C) - Ambient/vehicle temperature
- vehicle_speed (float, mph) - Current vehicle speed

# Enhanced DTC Data
- dtc_code (string) - Already have, keep as is
- dtc_subsystem (string) - Subsystem affected (Battery, Engine, Brakes, etc.)
- dtc_severity (string) - LOW, MEDIUM, HIGH, CRITICAL
- dtc_recommendation (string) - Action recommendation (e.g., "Replace hybrid battery pack")
- dtc_explanation (string) - Detailed explanation of the fault
- dtc_timestamp (datetime) - When DTC was first logged

# Anomalies
- anomaly_type (string) - SOC drop, High temperature, DTC logged, Service event
- anomaly_timestamp (datetime) - When anomaly occurred
- anomaly_severity (string) - INFO, WARNING, CRITICAL
- anomaly_description (string) - Human-readable description
- anomaly_icon (string) - Icon type (lightning, thermometer, wrench, X, etc.)

# Optional: Calculated Health Scores
- overall_health_score (float, 0-100) - Composite health score (can be calculated on-the-fly or stored)
- health_score_components (string/JSON) - Breakdown by subsystem
```

### Inference Log CSV (No changes needed):
```
# Keep existing columns only:
- timestamp (datetime) - Join key with historical.telematics_timestamp
- vin (string) - Join key with historical.vin
- model (string)
- primary_failed_part (string)
- mileage (float) - Prediction input/feature (used by ML model)
- age (float) - Prediction input/feature (used by ML model)
- pred_prob (float) - Prediction output
- pred_prob_pct (float) - Prediction output
- lat (float) - Location data (may be used as feature)
- lon (float) - Location data (may be used as feature)
- prescriptive_summary (string) - Prediction output
```

**Important Notes**:
1. **Inference Log contains**: Prediction outputs + prediction inputs/features (like `mileage`, `age`) that were used by the ML model
2. **Inference Log does NOT contain**: Sensor readings (like `battery_soc`, `battery_voltage`, `coolant_temperature`) - these come from telematics and belong in Historical CSV
3. **Key Distinction**: 
   - `mileage`/`age` = **Features** used for prediction (can be aggregated/bucketed)
   - `battery_soc`/`battery_voltage` = **Sensor readings** from telematics (real-time measurements)
4. All telematics/sensor data comes from Historical CSV via join

---

## Join Logic for Digital Twin View

### Primary Join Strategy:
```python
# Join Historical CSV with Inference Log CSV
df_digital_twin = df_inference_log.merge(
    df_historical,
    left_on=['vin', 'timestamp'],
    right_on=['vin', 'telematics_timestamp'],
    how='left',  # Keep all inference records, add telematics data when available
    suffixes=('_inference', '_telematics')
)

# Alternative: Time window join (if exact timestamp match not available)
# Match within ±1 hour window
df_digital_twin = df_inference_log.merge_asof(
    df_historical.sort_values('telematics_timestamp'),
    left_on='timestamp',
    right_on='telematics_timestamp',
    by='vin',
    direction='nearest',
    tolerance=pd.Timedelta('1h')
)
```

### Join Keys:
- **Primary**: `vin` (exact match)
- **Secondary**: `timestamp` (inference_log) ≈ `telematics_timestamp` (historical)
  - Exact match preferred
  - Time window fallback: ±1 hour, same day, or nearest timestamp

### Data Flow:
1. User selects a VIN from Inference Log
2. Query Historical CSV for all telematics records for that VIN
3. Match by timestamp (exact or nearest within window)
4. Combine prediction data (from Inference Log) with sensor data (from Historical)
5. Display unified Digital Twin view

---

## Data Generation Strategy

### For Historical CSV (Telematics Data):
1. **Generate telematics records** for each VIN with realistic timestamps
   - One row per telematics transmission (could be multiple per day)
   - Include `telematics_timestamp` for joining with inference log

2. **Sensor Readings**: Generate realistic ranges based on vehicle state
   - Battery SOC: 20-100% (normal), <20% (low), >90% (high)
   - Battery Voltage: 300-400V for EVs, 12-14V for ICE
   - Temperature: 60-100°F ambient, 150-220°F coolant
   - RPM: 0-6000 RPM (idle to max)

3. **Component Health**: Calculate based on sensor readings and DTC codes
   - Normal: All sensors in range, no DTCs
   - Warning: Some sensors out of range, minor DTCs
   - Critical: Multiple sensors critical, major DTCs

4. **Anomalies**: Generate based on sensor spikes, DTC occurrences, service events
   - SOC drop: Battery SOC decreases >10% in short time
   - High temperature: Coolant/ambient temp exceeds threshold
   - DTC logged: New DTC code appears
   - Service event: Maintenance/service performed

5. **DTC Data**: Enhance existing DTC codes with subsystem, severity, recommendations
   - Map DTC codes to subsystems (P0xxx = Battery, B1xxx = Body, etc.)
   - Assign severity based on DTC type
   - Generate recommendations based on DTC code

### For Inference Log CSV:
- **No changes needed** - Continue generating prediction outputs as before
- Ensure `timestamp` and `vin` are present for joining

---

## Implementation Priority

### Phase 1 (Minimum Viable):
- **Historical CSV**: Add `telematics_timestamp`, `model_year`, `battery_soc`, `battery_voltage`, `coolant_temperature`, `vehicle_speed`
- **Historical CSV**: Enhance DTC with `dtc_subsystem`, `dtc_severity`, `dtc_recommendation` (basic)
- **Join Logic**: Implement VIN + timestamp join
- **Digital Twin View**: Vehicle Overview, Health Score (derived from `pred_prob_pct`), Key Sensors, DTC Panel

### Phase 2 (Enhanced):
- **Historical CSV**: Add all component-specific sensor readings and health statuses
- **Historical CSV**: Add anomaly tracking columns
- **Digital Twin View**: Component Health cards, Anomalies Timeline, Advanced Sensors

### Phase 3 (Complete):
- Historical Trends: Component health trends over time
- Predictive Analytics: Component failure predictions
- Advanced Recommendations: AI-generated maintenance recommendations

---

## Notes

1. **Data Separation**: 
   - **Inference Log CSV** = Prediction outputs only (from ML model)
   - **Historical CSV** = All telematics/sensor data (from vehicle)
   - **Digital Twin View** = Joins both datasets for comprehensive view

2. **Join Strategy**:
   - Primary keys: `vin` (exact match) + `timestamp` (exact or nearest within window)
   - Use `telematics_timestamp` in Historical CSV to match with `timestamp` in Inference Log
   - Consider time window joins (±1 hour) if exact timestamp match is not always available

3. **Health Score Calculation**:
   - Primary driver: `pred_prob_pct` from Inference Log
   - Secondary factors: Sensor readings and DTC codes from Historical CSV
   - Can be calculated on-the-fly during join or pre-calculated and stored

4. **DTC Subsystem Mapping**: 
   - Create a lookup table mapping DTC codes to subsystems (P0xxx = Battery/Powertrain, B1xxx = Body, C1xxx = Chassis, U0xxx = Network)
   - Can be derived programmatically or stored as `dtc_subsystem` column

5. **Anomaly Detection**: 
   - Can be calculated on-the-fly from sensor readings (e.g., detect SOC drop by comparing consecutive readings)
   - Or stored as separate rows/events in Historical CSV with `anomaly_type`, `anomaly_timestamp`, etc.

6. **Data Volume Considerations**: 
   - Adding all sensor readings will significantly increase Historical CSV size
   - Consider data retention policies (e.g., keep detailed sensor data for last 90 days, aggregate older data)
   - Multiple telematics records per day per VIN is realistic (e.g., hourly or event-based transmissions)

7. **Timestamp Alignment**:
   - Ensure `telematics_timestamp` in Historical CSV aligns with `timestamp` in Inference Log
   - For synthetic data generation: Generate telematics records with timestamps that match inference log timestamps
   - In production: Both should use the same time source (vehicle telematics timestamp)

