# Dashboard Page Prescriptive Summary - Arrival Logic

This document describes the complete flow of how the prescriptive summary arrives at and is displayed on the **Dashboard page** (Prediction interface, Column 2).

---

## Table of Contents

1. [Overview](#overview)
2. [Dashboard Page Context](#dashboard-page-context)
3. [Complete Data Flow](#complete-data-flow)
4. [Step-by-Step Process](#step-by-step-process)
5. [Decision Logic](#decision-logic)
6. [Rendering Process](#rendering-process)

---

## Overview

The **Dashboard page** (also referred to as the Prediction interface) displays:
- **Column 1**: Historical Data Analysis
- **Column 2**: Predictive & Prescriptive Analysis + Chat
- **Column 3**: Inference Log Table + Severity Map

The prescriptive summary appears in **Column 2** and provides actionable insights based on ML predictions and historical data analysis.

**Key Functions**:
- `render_col2()` in `app.py` - Main orchestrator for Column 2
- `render_prescriptive_section()` in `components_col2.py` - Renders the prescriptive summary section
- `render_summary_ui()` in `app.py` - Generates the summary HTML content

---

## Dashboard Page Context

**Page Location**: Main dashboard (default page when app loads)

**Column 2 Components** (in order):
1. Predictive Analysis Controls (refresh interval, threshold settings)
2. Vehicle Information & KPI Display
3. **Prescriptive Summary** (this document's focus)
4. Vehicle Location & Nearest Dealers Map
5. Chat Interface

**Visual Layout**:
```
┌─────────────────────────────────────────────────────┐
│ Column 2: Predictive & Prescriptive Analysis       │
├─────────────────────────────────────────────────────┤
│ [Predictive Controls]                               │
├─────────────────────────────────────────────────────┤
│ Vehicle Info + KPI                                  │
├─────────────────────────────────────────────────────┤
│ ┌──────────────────┐  ┌──────────────────┐        │
│ │ Prescriptive     │  │ Location &       │        │
│ │ Summary          │  │ Dealers Map      │        │
│ │                  │  │                  │        │
│ └──────────────────┘  └──────────────────┘        │
├─────────────────────────────────────────────────────┤
│ Chat Interface                                       │
└─────────────────────────────────────────────────────┘
```

---

## Complete Data Flow

```
1. User loads Dashboard page
   ↓
2. render_col2() is called (Column 2 fragment)
   ↓
3. Generate prediction from historical data
   ↓
4. Fetch nearest dealers based on vehicle location
   ↓
5. Call render_prescriptive_section()
   ↓
6. Call render_summary_ui() with prediction data
   ↓
7. Generate enhanced prescriptive summary
   ↓
8. Display HTML summary in dashboard
   ↓
9. Log inference with summary to CSV
```

---

## Step-by-Step Process

### Step 1: Page Load & Initialization

**Function**: `render_col2()` in `app.py` (line 1757)

```python
@st.fragment
def render_col2():
    """
    Render Column 2: Predictive & Prescriptive Analysis + Chat.
    """
    # Load ML model
    model_pipe = load_model(MODEL_PATH)
    
    # Card container for predictive analysis
    st.markdown('<div class="card">', unsafe_allow_html=True)
```

**What Happens**:
- Streamlit fragment loads (enables auto-refresh)
- ML model is loaded from file
- HTML card container starts

---

### Step 2: Render Predictive Controls

**Function**: `render_predictive_controls()` (called at line 1774)

```python
refresh_interval = render_predictive_controls()
st_autorefresh(interval=refresh_interval * 1000, key="predictive_autorefresh")
```

**What Happens**:
- User can adjust refresh interval
- Auto-refresh is configured based on user selection
- Controls include threshold settings stored in `st.session_state['predictive_threshold_pct']`

---

### Step 3: Generate Prediction

**Function**: `generate_prediction()` in `components_col2.py` (line 116)

```python
inf_row, pred_prob = generate_prediction(df_history, model_pipe)
```

**Inputs**:
- `df_history`: Historical data DataFrame
- `model_pipe`: Loaded ML model pipeline

**Process**:
1. Samples a random row from historical data
2. Extracts features (model, part, mileage, age, lat, lon)
3. Runs ML model prediction
4. Returns prediction probability (0.0 to 1.0)

**Outputs**:
- `inf_row`: Dictionary with vehicle information and features
  - Keys: `model`, `primary_failed_part`, `mileage`, `age`, `lat`, `lon`, `vin`
- `pred_prob`: Float value (0.0 to 1.0) representing failure probability

**Example**:
```python
inf_row = {
    'model': 'Sentra',
    'primary_failed_part': 'Engine Cooling System',
    'mileage': 10200.0,
    'age': 0.5,
    'lat': 38.4405,
    'lon': -122.7144,
    'vin': '1N4AAPA1902317'
}
pred_prob = 0.808  # 80.8%
```

---

### Step 4: Fetch Nearest Dealers

**Function**: `fetch_nearest_dealers()` in `helper.py` (called at line 1792)

```python
if inf_row and pred_prob is not None:
    current_lat = inf_row.get("lat", 38.4405)
    current_lon = inf_row.get("lon", -122.7144)
    
    nearest_dealers, from_aws = fetch_nearest_dealers(
        current_lat=current_lat,
        current_lon=current_lon,
        place_index_name=constants['PLACE_INDEX_NAME'],
        aws_region=constants['AWS_REGION'],
        fallback_dealers=None,
        text_query="Nissan Service Center",
        top_n=3,
    )
```

**Purpose**: Find nearest authorized service centers for dealer recommendations

**Process**:
1. Extracts vehicle location (latitude, longitude) from `inf_row`
2. Queries AWS Place Index (or fallback) for nearby Nissan service centers
3. Returns top 3 nearest dealers with distance and ETA

**Output Structure**:
```python
nearest_dealers = [
    {
        'name': 'Vancouver Nissan Service Center',
        'distance_km': 12.5,
        'distance_miles': 7.8,
        'eta_min': 15,
        'lat': 49.2827,
        'lon': -123.1207
    },
    ...
]
```

**Why This Step Happens Early**: Dealer data is needed for prescriptive summary generation but must be fetched outside column context to avoid Streamlit nesting issues.

---

### Step 5: Create Layout Columns

**Function**: `render_col2()` (line 1808)

```python
pred_col, map_col = st.columns([1.5, 1], gap="medium")
```

**Layout**:
- `pred_col` (60% width): Contains vehicle info, KPI, and prescriptive summary
- `map_col` (40% width): Contains location map with dealers

---

### Step 6: Render Vehicle Info & KPI

**Function**: `render_col2()` (lines 1810-1858)

**Displayed Information**:
- Model name
- Primary failed part
- Mileage (formatted)
- Age (formatted)
- **Predicted Claim Probability** (large KPI display)

**KPI Color Logic**:
```python
threshold_pct = st.session_state.get('predictive_threshold_pct', config.model.default_threshold_pct)
kpi_color = config.colors.nissan_red if pred_prob * 100 >= threshold_pct else '#10b981'
```

- **Red** (#C3002F): If prediction ≥ threshold (high risk)
- **Green** (#10b981): If prediction < threshold (low risk)

---

### Step 7: Render Prescriptive Section

**Function**: `render_prescriptive_section()` in `components_col2.py` (line 351)

**Call**:
```python
if inf_row and pred_prob is not None and nearest_dealers is not None:
    _, prescriptive_summary_html = render_prescriptive_section(
        inf_row, 
        pred_prob, 
        render_summary_ui
    )
```

**Condition**: All three must be available:
- `inf_row`: Vehicle/prediction data exists
- `pred_prob`: Prediction probability is valid
- `nearest_dealers`: Dealer data fetched successfully

---

### Step 8: Inside `render_prescriptive_section()`

#### 8a. Render Section Header

```python
st.markdown(
    '<div class="card-header" style="height:34px; display:flex; align-items:center;">Prescriptive Summary</div>',
    unsafe_allow_html=True
)
```

#### 8b. Fetch Dealers (if not already available)

```python
current_lat = inf_row.get("lat", 38.4405)
current_lon = inf_row.get("lon", -122.7144)

nearest, from_aws = fetch_nearest_dealers(
    current_lat=current_lat,
    current_lon=current_lon,
    place_index_name=constants['PLACE_INDEX_NAME'],
    aws_region=constants['AWS_REGION'],
    fallback_dealers=None,
    text_query="Nissan Service Center",
    top_n=3,
)
```

**Note**: Dealers are fetched again here as a backup if they weren't fetched in Step 4.

#### 8c. Prepare Data for Summary Generation

**POC Mode** (if `constants['IS_POC']` is True):
```python
mock_dealer = {
    'name': 'United Nissan Dealer Service Center',
    'distance_km': round(19 * MILES_TO_KM_FACTOR, 2),
    'distance_miles': 19,
    'eta_min': 5
}

summary_html = render_summary_ui_func(
    'Sentra',
    'Engine Cooling System',
    '10200 miles',
    '6 months',
    80.8,
    mock_dealer
)
```

**Normal Mode**:
```python
mileage_display = f"{inf_row['mileage']:,.0f} miles"
age_display = f"{inf_row['age']:.1f} years"

nearest_dealer = nearest[0] if nearest else None

summary_html = render_summary_ui_func(
    inf_row['model'],
    inf_row['primary_failed_part'],
    mileage_display,
    age_display,
    pred_prob * 100,  # Convert to percentage
    nearest_dealer,
    vehicle_context=inf_row
)
```

**Key Parameters Passed**:
- `model_name`: Vehicle model
- `part_name`: Primary failed part
- `mileage_bucket`: Formatted mileage string
- `age_bucket`: Formatted age string
- `claim_pct`: Prediction probability as percentage (0-100)
- `nearest_dealer`: First dealer from list (or None)
- `vehicle_context`: Full inference row (for cohort analysis)

---

### Step 9: Generate Summary via `render_summary_ui()`

**Function**: `render_summary_ui()` in `app.py` (line 282)

**Entry Point Decision Logic**:

```python
default_threshold = config.model.default_threshold_pct  # Typically 30%

if claim_pct >= int(st.session_state.get('predictive_threshold_pct', default_threshold)):
    # Use enhanced prescriptive summary
    summary_html = generate_enhanced_prescriptive_summary(...)
else:
    # Use simple hardcoded summary
    risk_token = calculate_risk_level(claim_pct)
    summary_html = f"<strong>{risk_token} risk ({claim_pct}%)</strong>: Continue routine monitoring..."
```

**Decision Tree**:

```
Is claim_pct >= threshold? (e.g., ≥ 30%)
├─ YES → Generate enhanced prescriptive summary
│   ├─ Analyze historical trends
│   ├─ Identify problematic suppliers
│   ├─ Extract failure reasons
│   ├─ Include dealer recommendations
│   └─ Include cohort insights
│
└─ NO → Show simple status message
    └─ "Low risk (X%): Continue routine monitoring..."
```

---

### Step 10: Enhanced Prescriptive Summary Generation

**Function**: `generate_enhanced_prescriptive_summary()` in `helper.py` (line 1414)

**Data Sources Used**:

1. **Historical Data (`df_history`)**:
   - Filtered by `model` and `primary_failed_part`
   - Used for trend analysis, supplier analysis, failure reasons

2. **Prediction Data (`inf_row`)**:
   - `claim_pct`: Risk level determination
   - `vehicle_context`: For cohort analysis (VIN, manufacturing_date)

3. **Dealer Data (`nearest_dealer`)**:
   - Name, distance, ETA for recommendations

**Analysis Functions Called**:

1. **Trend Analysis**:
   ```python
   trend_info = get_model_trend_info(model_data, model_name, part_name)
   ```
   - Analyzes failure trends over time
   - Determines if failures are rising, declining, or stable

2. **Supplier Analysis**:
   ```python
   supplier_info = get_supplier_analysis(model_data, part_name)
   ```
   - Identifies problematic suppliers
   - Recommends better alternatives

3. **Failure Reasons**:
   ```python
   failure_reasons = get_failure_reasons(model_data, part_name)
   ```
   - Extracts top failure modes
   - Calculates percentages

4. **Cohort Insight**:
   ```python
   cohort_insight = _build_cohort_insight(vehicle_context, df_history, model_name, part_name)
   ```
   - Identifies production batch issues
   - Shows how many vehicles in same cohort have same problem

5. **Dealer Recommendation**:
   ```python
   dealer_info = get_dealer_recommendation(nearest_dealer)
   ```
   - Formats dealer information for display

**Summary Content Assembly** (in priority order):

1. **Opening Statement with Trend**
   - Always included
   - Uses randomized language patterns
   - Includes trend direction and statistics

2. **Dealer Recommendation**
   - **Condition**: `claim_pct >= 40%` AND `dealer_info` available
   - Includes dealer name, distance, ETA

3. **Supplier Analysis**
   - **Condition**: `supplier_info['has_supplier_data'] == True`
   - Identifies problematic suppliers
   - Recommends alternatives (if available)

4. **Failure Reasons**
   - **Condition**: `failure_reasons` list is not empty
   - Shows top 1-2 failure modes with percentages

5. **Risk Assessment**
   - Always included
   - **High Risk** (`≥ 75%`): Urgent technical action required
   - **Moderate Risk** (`40-74%`): Preventive maintenance advised
   - **Low Risk** (`< 40%`): Routine monitoring sufficient

6. **Closing Statement**
   - **Condition**: `len(summary_parts) >= 3`
   - Varies based on risk level
   - Includes cohort coordination (if applicable)

---

### Step 11: Summary Display Formatting

**Function**: `render_summary_ui()` continues (lines 330-350)

```python
split_html = re.split(r'\n\s*\n', summary_html, maxsplit=1)
first_para_html = split_html[0].strip()
```

**Process**:
1. Split summary into first paragraph and remaining content
2. Display first paragraph prominently
3. Show remaining content in expandable section

**Display Structure**:
```html
<div style="padding: 16px;">
    <!-- First paragraph (always visible) -->
    <div>{first_para_html}</div>
    
    <!-- Expandable section for full summary -->
    <details>
        <summary>Show full summary</summary>
        <div>{remaining_html}</div>
    </details>
</div>
```

**Why Split**: First paragraph provides quick insight, full summary available on demand for detailed analysis.

---

### Step 12: Log Inference with Summary

**Function**: `render_col2()` (lines 1882-1887)

```python
if inf_row and pred_prob is not None:
    summary_plain = None
    if prescriptive_summary_html:
        summary_plain, _, _, _ = _extract_plain_and_bullets(prescriptive_summary_html)
    log_inference(inf_row, pred_prob, prescriptive_summary=summary_plain)
```

**Process**:
1. Extract plain text from HTML summary (remove formatting)
2. Append inference log entry to CSV
3. Includes all prediction data + plain text summary

**Logged Data**:
```python
{
    'timestamp': '2025-01-XX XX:XX:XX',
    'vin': '1N4AAPA1902317',
    'model': 'Sentra',
    'primary_failed_part': 'Engine Cooling System',
    'mileage': 10200.0,
    'age': 0.5,
    'pred_prob': 0.808,
    'pred_prob_pct': 80.8,
    'prescriptive_summary': 'Telemetry analysis reveals...',
    'lat': 38.4405,
    'lon': -122.7144,
    ...
}
```

---

## Decision Logic

### Summary Type Selection

```python
threshold = st.session_state.get('predictive_threshold_pct', default_threshold)

if claim_pct >= threshold:
    → Enhanced Prescriptive Summary
else:
    → Simple Status Message
```

**Enhanced Summary Includes**:
- Trend analysis
- Supplier insights
- Failure root causes
- Dealer recommendations
- Cohort insights
- Risk-based recommendations

**Simple Summary Includes**:
- Risk level badge
- Prediction percentage
- Generic monitoring message

---

### Dealer Recommendation Inclusion

```python
if dealer_info and claim_pct >= 40:
    → Include dealer recommendation
else:
    → Skip dealer recommendation
```

**Rationale**: Only recommend dealer visits for moderate+ risk scenarios (≥40%)

---

### Cohort Insight Inclusion

```python
if claim_pct >= 40 and cohort_insight exists:
    → Include cohort-specific message
else:
    → Use standard moderate risk message
```

**Cohort Insight Shows**:
- Number of vehicles in same production batch with same issue
- Manufacturing plant and month
- Coordination recommendations

---

## Rendering Process

### HTML Generation

The prescriptive summary is generated as **HTML-formatted text** and displayed using Streamlit's `st.markdown()` with `unsafe_allow_html=True`.

**Format**:
```html
<div style='text-align: justify; line-height: 1.6;'>
    <p style='margin-bottom: 12px;'>{paragraph 1}</p>
    <p style='margin-bottom: 12px;'>{paragraph 2}</p>
    ...
</div>
```

### Visual Styling

**Styled Elements**:
- **Bold highlights**: Model names, part names, dealer names, failure modes
- **Color coding**: Risk assessment sections (red/amber/green)
- **Paragraph spacing**: 12px margin between paragraphs
- **Justified text**: Professional appearance

### Responsive Display

**Two-Level Display**:
1. **First Paragraph**: Always visible (quick insight)
2. **Full Summary**: Expandable section (detailed analysis)

**User Interaction**:
- Click "Show full summary" to expand
- Click "Hide full summary" to collapse

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────┐
│ 1. Dashboard Page Loads (render_col2)                   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Generate Prediction (generate_prediction)            │
│    Input: df_history, model_pipe                        │
│    Output: inf_row, pred_prob                           │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Fetch Nearest Dealers (fetch_nearest_dealers)        │
│    Input: lat, lon                                       │
│    Output: nearest_dealers list                         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Render Prescriptive Section                          │
│    (render_prescriptive_section)                        │
│    Input: inf_row, pred_prob, render_summary_ui         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Generate Summary UI (render_summary_ui)              │
│    Decision: claim_pct >= threshold?                    │
└─────────────────┬───────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
┌──────────────┐   ┌──────────────────┐
│ Simple       │   │ Enhanced         │
│ Summary      │   │ Summary          │
│              │   │                  │
│ - Risk badge │   │ - Trend analysis │
│ - Status msg │   │ - Supplier info  │
│              │   │ - Failure reasons│
│              │   │ - Dealer recs    │
│              │   │ - Cohort insights│
└──────────────┘   └──────────────────┘
        │                   │
        └─────────┬─────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 6. Display HTML Summary in Dashboard                    │
│    (Streamlit markdown rendering)                       │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 7. Log Inference with Summary (log_inference)           │
│    Appends to inference_log.csv                         │
└─────────────────────────────────────────────────────────┘
```

---

## Key Configuration Values

### Threshold Settings

- **Default Threshold**: `config.model.default_threshold_pct` (typically 30%)
- **User Override**: `st.session_state['predictive_threshold_pct']`
- **Dealer Recommendation Threshold**: 40% (hardcoded)
- **High Risk Threshold**: 75% (hardcoded)

### Display Settings

- **Summary Position**: Left column (60% width)
- **First Paragraph**: Always visible
- **Full Summary**: Expandable section
- **Auto-refresh**: Configurable interval (default: 30 seconds)

---

## Error Handling & Fallbacks

### Fallback Chain

```
1. Enhanced Prescriptive Summary (generate_enhanced_prescriptive_summary)
   ↓ (if fails)
2. Basic Enhanced Summary (generate_basic_enhanced_summary)
   ↓ (if fails)
3. Bedrock LLM Summary (get_bedrock_summary)
   ↓ (if fails)
4. Hardcoded Simple Message
```

### Missing Data Handling

- **No Historical Data**: Falls back to basic summary
- **No Dealer Data**: Summary generated without dealer recommendations
- **No Prediction Data**: Shows placeholder "Waiting for prediction data..."
- **Empty Summary**: Error logged, fallback message shown

---

## Example: Complete Flow

### Scenario: High-Risk Prediction (85%)

**Input Data**:
- Model: "Leaf"
- Part: "Battery"
- Mileage: 45,230 miles
- Age: 3.2 years
- Prediction: 85%
- Location: Vancouver, BC (lat: 49.2827, lon: -123.1207)

**Step-by-Step Result**:

1. **Prediction Generated**: `pred_prob = 0.85`

2. **Dealers Fetched**: 3 nearest Nissan service centers found

3. **Summary Type**: Enhanced (85% > 30% threshold)

4. **Historical Analysis**:
   - Trend: Strong rising (increasing battery failures)
   - Supplier: BatteryCo identified as problematic (18% failure rate)
   - Failure Reason: "Premature degradation" (58% of cases)
   - Cohort: 12 vehicles from same production batch affected

5. **Summary Generated**:
   ```
   Paragraph 1 (Visible):
   "Telemetry analysis reveals an escalating pattern of battery failures 
   in the Leaf model over the past 6 months (avg: 12.3/month, latest: 18)."
   
   Full Summary (Expandable):
   - Dealer recommendation: "Vancouver Nissan Service Center, 7.8 mi away"
   - Supplier analysis: BatteryCo issues + PowerCell recommendation
   - Failure reasons: Premature degradation details
   - Risk assessment: "High claim likelihood: 85% probability indicates..."
   - Cohort insight: "12 vehicles from same batch affected..."
   - Closing: "Immediate technical escalation required..."
   ```

6. **Displayed**: In left column of dashboard, below vehicle info/KPI

7. **Logged**: Inference entry created in `inference_log.csv` with full summary

---

## Notes

- **Auto-refresh**: Column 2 uses `@st.fragment` decorator for auto-refresh capability
- **State Management**: Threshold settings persist in `st.session_state`
- **POC Mode**: Special mode with mock data for demonstration purposes
- **Error Resilience**: Multiple fallback levels ensure summary always available
- **Performance**: Dealer fetching happens early to avoid blocking summary generation

---

## Version History

- **2025-01-XX**: Initial documentation
- Logic based on `app.py` and `components_col2.py` implementation

