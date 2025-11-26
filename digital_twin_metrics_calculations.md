# Digital Twin Metrics Calculation Logic

This document describes the calculation logic used for each metrics card displayed on the Digital Twin dashboard page.

---

## Table of Contents

1. [Overall Health Score Gauge](#1-overall-health-score-gauge)
2. [Key Sensor Snapshot](#2-key-sensor-snapshot)
3. [Component Health Cards](#3-component-health-cards)
   - [Battery Component](#31-battery-component)
   - [Fuel System Component](#32-fuel-system-component)
   - [Engine Component](#33-engine-component)
   - [Brakes Component](#34-brakes-component)
4. [DTC Fault Panel](#4-dtc-fault-panel)

---

## 1. Overall Health Score Gauge

**Location:** Top-right panel  
**Function:** `calculate_health_score()`  
**Range:** 0-100 points  
**Status Levels:** Healthy (≥70), Warning (40-69), Critical (<40)

### Calculation Formula

```
base_score = 100 - pred_prob_pct
```

Where:
- `pred_prob_pct` = Prediction probability percentage (0-100) from ML model inference
- Higher prediction probability = Lower health score (inverse relationship)

### Adjustments

#### Battery State of Charge (SOC) Penalties
- If `battery_soc < 20%`: Subtract 20 points
- If `20% ≤ battery_soc < 50%`: Subtract 10 points
- If `battery_soc ≥ 50%`: No penalty

#### DTC Severity Penalties
- `CRITICAL`: Subtract 30 points
- `HIGH`: Subtract 20 points
- `MEDIUM`: Subtract 10 points
- `LOW`: Subtract 5 points

### Final Score Calculation

```python
health_score = max(0, min(100, base_score + adjustments))
```

The score is clamped to the range [0, 100].

### Status Determination

| Health Score Range | Status |
|-------------------|--------|
| ≥ 70 | Healthy |
| 40 - 69 | Warning |
| < 40 | Critical |

### Example

**Input:**
- `pred_prob_pct` = 60%
- `battery_soc` = 45%
- `dtc_severity` = "HIGH"

**Calculation:**
1. `base_score = 100 - 60 = 40`
2. Battery adjustment: `-10` (45% is between 20-50%)
3. DTC adjustment: `-20` (HIGH severity)
4. `health_score = max(0, min(100, 40 - 10 - 20)) = 10`
5. Status: **Critical** (< 40)

---

## 2. Key Sensor Snapshot

**Location:** Middle panel  
**Function:** `render_sensor_snapshot()`  
**Purpose:** Displays real-time sensor readings with visual indicators

### Sensor Selection Logic

The displayed sensors vary based on vehicle type (EV vs. Non-EV):

#### Electric Vehicles (Leaf, Ariya)
1. **Battery SOC** - State of Charge (%)
   - Good indicator: `> 50%`
2. **Battery Voltage** (V)
   - Good indicator: `300-400V` (EV) or `11-14V` (12V battery)
3. **Ambient Temp** (°F)
   - Always displayed (no good/bad threshold)
4. **Battery Temp** (°F)
   - Conversion: `°F = (°C × 9/5) + 32`
   - Good indicator: `68-113°F` (20-45°C)
5. **Motor RPM** (rpm)
   - Always displayed (no good/bad threshold)
6. **Coolant Temp** (°F)
   - Good indicator: `180-210°F`
7. **Speed** (mph)
   - Always displayed (no good/bad threshold)

#### Non-Electric Vehicles
1. **Engine Coolant Temp** (°F)
   - Good indicator: `180-210°F`
2. **Oil Pressure** (psi)
   - Good indicator: `25-80 psi`
3. **Engine RPM** (rpm)
   - Always displayed (no good/bad threshold)
4. **Fuel Level** (%)
   - Uses `battery_soc` field as fuel level indicator
   - Good indicator: `> 30%`
5. **Engine Load** (%)
   - Good indicator: `< 90%`
6. **Speed** (mph)
   - Always displayed (no good/bad threshold)

### Visual Indicators

- **Green color**: Sensor value within optimal range (`is_good = True`)
- **Muted color**: Sensor value outside optimal range or no threshold defined

---

## 3. Component Health Cards

**Location:** Bottom section (3 cards)  
**Function:** `render_component_health_card()`  
**Purpose:** Display health status and metrics for individual vehicle components

### Overall Component Health Calculation

For each component, individual metric health scores are calculated, then averaged:

```python
health_scores = [get_metric_health(metric_name, value, component) for each metric]
overall_health = sum(health_scores) / len(health_scores)
```

### Component Status Determination

| Overall Health | Status |
|--------------|--------|
| ≥ 80 | Normal |
| 50 - 79 | Warning |
| < 50 | Critical |

---

### 3.1 Battery Component

**Metrics Displayed:**
- Temperature (°C)
- Voltage (V)

#### Temperature Health Score

| Temperature Range | Health Score |
|------------------|--------------|
| 20-40°C | 100% (Optimal) |
| 0-20°C or 40-60°C | 50% (Moderate) |
| < 0°C or > 60°C | 0% (Critical) |

#### Voltage Health Score

| Voltage Range | Health Score |
|--------------|--------------|
| 300-400V | 100% (Optimal) |
| 250-300V or 400-450V | 60% (Warning) |
| < 250V or > 450V | 30% (Critical) |

**Note:** For 12V batteries, the voltage range is typically 11-14V, but the same scoring logic applies.

#### Status Text Logic

- **Normal**: "No major issues detected"
- **Warning/Critical**: 
  - If `voltage < 300V`: "Voltage below optimal range"
  - Otherwise: "Capacity degradation detected"

---

### 3.2 Fuel System Component

**Metrics Displayed:**
- Fuel Level (%)
- Oil Pressure (psi)
- Engine Load (%)

#### Fuel Level Health Score

| Fuel Level | Health Score |
|-----------|--------------|
| > 50% | 100% (Optimal) |
| 20-50% | 60% (Warning) |
| < 20% | 20% (Critical) |

#### Oil Pressure Health Score

| Oil Pressure | Health Score |
|-------------|--------------|
| 25-80 psi | 100% (Optimal) |
| 15-25 psi or 80-100 psi | 60% (Warning) |
| < 15 psi or > 100 psi | 30% (Critical) |

#### Engine Load Health Score

| Engine Load | Health Score |
|------------|--------------|
| < 70% | 100% (Optimal) |
| 70-90% | 60% (Warning) |
| > 90% | 30% (Critical) |

#### Status Text Logic

Priority order:
1. If `fuel_level < 20%`: "Low fuel level detected"
2. Else if `oil_pressure < 25 psi`: "Low oil pressure detected"
3. Else if `engine_load > 90%`: "High engine load detected"
4. Else if status is Normal: "Fuel system operating within normal parameters"
5. Otherwise: "Fuel system requires attention"

---

### 3.3 Engine Component

**Metrics Displayed:**
- Water Pump Speed (RPM)
- Coolant Temperature (°C)

#### Water Pump Speed Health Score

| Water Pump Speed | Health Score |
|-----------------|--------------|
| > 1500 RPM | 100% (Optimal) |
| 1000-1500 RPM | 70% (Warning) |
| < 1000 RPM | 40% (Critical) |

#### Coolant Temperature Health Score

| Coolant Temperature | Health Score |
|---------------------|--------------|
| 180-210°C | 100% (Optimal) |
| 160-180°C or 210-230°C | 60% (Warning) |
| < 160°C or > 230°C | 30% (Critical) |

#### Status Text Logic

Priority order:
1. If `water_pump < 1000 RPM`: "Water pump speed below optimal"
2. Else if `coolant_temp < 180°C`: "Coolant temperature below optimal"
3. Else if `coolant_temp > 210°C`: "Coolant temperature above optimal"
4. Else if status is Normal: "Engine operating within normal parameters"
5. Otherwise: "Engine requires attention"

---

### 3.4 Brakes Component

**Metrics Displayed:**
- Brake Pressure (bar)
- Brake Pad Wear (%)

#### Brake Pressure Health Score

| Brake Pressure | Health Score |
|---------------|--------------|
| 25-35 bar | 100% (Optimal) |
| 20-25 bar or 35-40 bar | 70% (Warning) |
| < 20 bar or > 40 bar | 40% (Critical) |

#### Brake Pad Wear Health Score

| Pad Wear | Health Score |
|----------|--------------|
| < 50% | 100% (Optimal) |
| 50-70% | 60% (Warning) |
| > 70% | 30% (Critical) |

#### Status Text Logic

Priority order:
1. If `pad_wear > 70%`: "Brake pad wear critical"
2. Else if `pad_wear > 50%`: "Brake pad wear moderate"
3. Else if `brake_pressure < 25 bar`: "Brake pressure below optimal"
4. Else if `brake_pressure > 35 bar`: "Brake pressure above optimal"
5. Else if status is Normal: "Brakes operating within normal parameters"
6. Otherwise: "Brakes require attention"

---

## 4. DTC Fault Panel

**Location:** Bottom-left panel  
**Function:** `render_dtc_panel()`  
**Purpose:** Display Diagnostic Trouble Code information

### Data Source Priority

1. **Primary**: DTC data from merged row (inference log + historical data)
2. **Fallback**: Most recent DTC from historical data for the given VIN

### Displayed Information

- **DTC Code**: e.g., "P0A7F"
- **Subsystem**: e.g., "Battery", "Engine", "Transmission"
- **Severity**: LOW, MEDIUM, HIGH, CRITICAL
- **Recommended Action**: Prescriptive text from DTC library
- **Explanation**: Detailed description of the fault

### DTC Generation Logic

DTCs are generated based on component health using `_generate_dtc_based_on_health()`:

#### Health Risk Score Calculation

```python
health_risk = 0.0

# Base risk from prediction probability (50% weight)
if pred_prob_pct is not None:
    health_risk += 0.5 * (pred_prob_pct / 100.0)

# Mileage contribution (25% weight, normalized to 150k miles)
if mileage is not None:
    mileage_risk = min(1.0, mileage / 150000.0)
    health_risk += 0.25 * mileage_risk

# Age contribution (15% weight, normalized to 15 years)
if age is not None:
    age_risk = min(1.0, age / 15.0)
    health_risk += 0.15 * age_risk
```

#### DTC Generation Probability

```python
base_probability = 0.10 + (0.40 * health_risk)  # 10% to 50% chance
```

- Minimum 10% chance of generating a DTC
- Maximum 50% chance (when health_risk = 1.0)
- DTC selection is biased towards higher severity codes for higher health risk

---

## Notes

### Data Sources

- **Inference Log**: Contains `pred_prob_pct`, `mileage`, `age`, `timestamp`
- **Historical Data**: Contains sensor readings (`battery_soc`, `coolant_temperature`, etc.), `telematics_timestamp`, `date`
- **Merged Data**: Joins inference log with historical data by VIN and timestamp proximity

### Missing Data Handling

- Missing sensor values display as "N/A"
- Missing metrics default to moderate health (75%) for component calculations
- Health scores are clamped to valid ranges to prevent negative or >100 values

### Model-Specific Logic

- **EV Models** (Leaf, Ariya): Display battery-focused metrics
- **Non-EV Models** (Sentra, etc.): Display fuel system and engine metrics
- Model determination uses `model_history` field as source of truth

---

## Version History

- **2025-01-XX**: Initial documentation
- Calculation logic based on `digital_twin_component.py` implementation

