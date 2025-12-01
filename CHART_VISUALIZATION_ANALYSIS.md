# Chart Visualization for Chat Results - Analysis & Implementation Plan

## Current Architecture

### How Chat Results Are Displayed
1. **Response Format**: Handlers return HTML strings
2. **Rendering**: HTML is displayed via `st.markdown(..., unsafe_allow_html=True)` in `app.py`
3. **Result Formatting**: `_format_results_simple()` converts DataFrame to HTML table
4. **Chart Libraries Available**: Plotly (plotly.express, plotly.graph_objects) is already used in the app

### Current Flow
```
User Query → QueryRouter → Handler → SQL Query → DataFrame Results → HTML String → st.markdown()
```

---

## Solution Approaches

### **Option 1: Plotly HTML Embedding (Recommended)**
**How it works:**
- Generate Plotly chart as HTML `<div>` using `plotly.offline.plot(fig, output_type='div', include_plotlyjs='cdn')`
- Embed the HTML div directly in the chat response HTML
- Streamlit's `st.markdown(unsafe_allow_html=True)` will render it

**Pros:**
- ✅ Simple implementation - no frontend changes needed
- ✅ Charts render inline with text
- ✅ Interactive (zoom, pan, hover)
- ✅ Works with existing HTML rendering

**Cons:**
- ⚠️ Larger HTML payload (includes Plotly.js CDN)
- ⚠️ Slightly slower initial render

**Implementation:**
```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

def _generate_chart_html(results: pd.DataFrame, chart_type: str) -> str:
    fig = px.bar(results, x='age_bucket', y='total_failures')
    chart_html = plot(fig, output_type='div', include_plotlyjs='cdn')
    return chart_html
```

---

### **Option 2: Separate Chart Rendering (Alternative)**
**How it works:**
- Return special marker in HTML: `<div class="chart-placeholder" data-chart-type="bar" data-chart-data='{json}'>`
- In `app.py`, parse HTML, extract chart data, render with `st.plotly_chart()`
- Replace placeholder with actual chart

**Pros:**
- ✅ Smaller HTML payload
- ✅ More control over chart rendering
- ✅ Can use Streamlit's native chart components

**Cons:**
- ⚠️ Requires frontend parsing logic
- ⚠️ More complex implementation
- ⚠️ Charts appear after text (not inline)

**Implementation:**
```python
# In handler:
chart_data = results.to_dict('records')
chart_marker = f'<div class="chart-placeholder" data-chart-type="bar" data-chart-data=\'{json.dumps(chart_data)}\'></div>'

# In app.py:
import re
chart_pattern = r'<div class="chart-placeholder"[^>]*data-chart-data=\'([^\']+)\'[^>]*></div>'
matches = re.findall(chart_pattern, html_response)
for match in matches:
    chart_data = json.loads(match)
    fig = px.bar(pd.DataFrame(chart_data), ...)
    st.plotly_chart(fig)
```

---

### **Option 3: Hybrid Approach (Best UX)**
**How it works:**
- Detect chart-friendly queries
- Generate both HTML text summary AND chart
- Return structured response: `{"html": "...", "chart": {...}}`
- In frontend, render HTML first, then chart below

**Pros:**
- ✅ Best user experience (text + chart)
- ✅ Flexible (can show both or just one)
- ✅ Clean separation of concerns

**Cons:**
- ⚠️ Requires response format change
- ⚠️ More complex handler return type

---

## Recommended Implementation: Option 1 (Plotly HTML Embedding)

### Why Option 1?
1. **Minimal Changes**: Works with existing HTML rendering
2. **No Frontend Changes**: All logic in handlers
3. **Interactive Charts**: Users can zoom, pan, hover
4. **Inline Display**: Charts appear naturally in conversation flow

---

## Implementation Plan

### Step 1: Create Chart Detection Logic
**File**: `chat/handlers_text_to_sql.py`

```python
def _is_chart_friendly_query(self, user_query: str, results: pd.DataFrame) -> bool:
    """
    Determine if query results should be displayed as a chart.
    
    Chart-friendly queries:
    - Breakdown queries: "by X", "per X", "grouped by X"
    - Time series: "over time", "by month", "by quarter", "by year"
    - Aggregations: "count by", "sum by", "average by"
    - Comparisons: "compare", "top N", "worst N"
    - Results with 2-20 rows (too few = single value, too many = table)
    """
    query_lower = user_query.lower()
    
    # Time series patterns
    time_patterns = [
        r'\b(by|per)\s+(month|quarter|year|week|day|time)',
        r'\b(over|across)\s+(time|months?|quarters?|years?)',
        r'\b(trend|trends|over time)',
    ]
    
    # Breakdown patterns
    breakdown_patterns = [
        r'\b(by|per|grouped by|group by)\s+\w+',
        r'\b(count|sum|total|average|avg)\s+(by|per)',
    ]
    
    # Comparison patterns
    comparison_patterns = [
        r'\b(top|worst|best|most|least)\s+\d+',
        r'\b(compare|comparison)',
    ]
    
    has_time_pattern = any(re.search(p, query_lower) for p in time_patterns)
    has_breakdown = any(re.search(p, query_lower) for p in breakdown_patterns)
    has_comparison = any(re.search(p, query_lower) for p in comparison_patterns)
    
    # Check result structure
    row_count = len(results)
    col_count = len(results.columns)
    
    # Ideal for charts: 2-20 rows, 2 columns (category + value)
    is_ideal_structure = (2 <= row_count <= 20) and (col_count == 2)
    
    # Also good: 2-50 rows with time/breakdown pattern
    is_breakdown_structure = (2 <= row_count <= 50) and (has_breakdown or has_time_pattern)
    
    return (has_time_pattern or has_breakdown or has_comparison) and (is_ideal_structure or is_breakdown_structure)
```

### Step 2: Create Chart Generation Function
**File**: `chat/handlers_text_to_sql.py`

```python
def _generate_chart_from_results(
    self, 
    results: pd.DataFrame, 
    user_query: str
) -> Optional[str]:
    """
    Generate Plotly chart HTML from query results.
    
    Returns:
        HTML string with embedded Plotly chart, or None if chart can't be generated
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.offline import plot
        
        query_lower = user_query.lower()
        row_count = len(results)
        col_count = len(results.columns)
        
        # Determine chart type based on query and data
        if col_count == 2:
            # Two columns: category + value
            x_col = results.columns[0]
            y_col = results.columns[1]
            
            # Check if y_col is numeric
            if pd.api.types.is_numeric_dtype(results[y_col]):
                # Determine chart type
                if any(word in query_lower for word in ['trend', 'over time', 'by month', 'by quarter', 'by year']):
                    # Time series - use line chart
                    fig = px.line(results, x=x_col, y=y_col, 
                                 title=f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}")
                elif row_count <= 10:
                    # Small dataset - bar chart
                    fig = px.bar(results, x=x_col, y=y_col,
                                title=f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}")
                else:
                    # Larger dataset - horizontal bar for readability
                    fig = px.bar(results, x=y_col, y=x_col, orientation='h',
                                title=f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}")
                
                # Style the chart
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#cfe9ff', size=12),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                # Generate HTML
                chart_html = plot(fig, output_type='div', include_plotlyjs='cdn', config={'displayModeBar': False})
                return chart_html
        
        # Multi-column results - could use grouped bar chart
        elif col_count > 2 and row_count <= 15:
            # Use first column as x-axis, others as series
            x_col = results.columns[0]
            numeric_cols = [col for col in results.columns[1:] if pd.api.types.is_numeric_dtype(results[col])]
            
            if numeric_cols:
                fig = go.Figure()
                for col in numeric_cols:
                    fig.add_trace(go.Bar(
                        name=col.replace('_', ' ').title(),
                        x=results[x_col],
                        y=results[col]
                    ))
                
                fig.update_layout(
                    barmode='group',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#cfe9ff', size=12),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                chart_html = plot(fig, output_type='div', include_plotlyjs='cdn', config={'displayModeBar': False})
                return chart_html
        
        return None
        
    except Exception as e:
        logger.warning(f"Chart generation failed: {e}")
        return None
```

### Step 3: Integrate Chart Generation into Response Flow
**File**: `chat/handlers_text_to_sql.py`

Modify `_generate_natural_language_response()`:

```python
def _generate_natural_language_response(...):
    # ... existing code ...
    
    # Check if results are chart-friendly
    if self._is_chart_friendly_query(user_query, results):
        chart_html = self._generate_chart_from_results(results, user_query)
        if chart_html:
            # Add chart to response
            response = f"{response}<div style='margin-top: 16px;'>{chart_html}</div>"
    
    return response
```

Also modify `_format_results_simple()`:

```python
def _format_results_simple(self, results: pd.DataFrame, user_query: str = "") -> str:
    # ... existing table generation code ...
    
    # Check if we should add a chart
    if self._is_chart_friendly_query(user_query, results):
        chart_html = self._generate_chart_from_results(results, user_query)
        if chart_html:
            return f"<p style='margin-bottom: 8px;'>{answer_text}</p>" \
                   f"<div style='margin-top: 12px; margin-bottom: 12px;'>{chart_html}</div>" \
                   f"<div style='margin-top: 12px;'>{html_table}</div>"
    
    # Return table only
    return f"<p style='margin-bottom: 8px;'>{answer_text}</p><div style='margin-top: 0;'>{html_table}</div>"
```

---

## Chart Types by Query Pattern

| Query Pattern | Chart Type | Example |
|--------------|-----------|---------|
| "by month/quarter/year" | Line Chart | "Failures by month" |
| "by X" (breakdown) | Bar Chart | "Failures by model" |
| "top N / worst N" | Bar Chart | "Top 5 failing parts" |
| "compare" | Grouped Bar | "Compare Leaf vs Sentra" |
| "over time" | Line Chart | "Failure trends over time" |
| "count by" | Bar Chart | "Vehicle count by city" |

---

## Frontend Display

### Current Rendering (No Changes Needed!)
The HTML with embedded Plotly chart will render automatically:

```python
# In app.py - _render_chat_html_and_scroll()
assistant_html = (m.get("text") or "").replace("\n", "<br/>")
# This already handles HTML, so Plotly divs will render!
```

**No frontend changes required!** The existing `st.markdown(..., unsafe_allow_html=True)` will render the Plotly charts.

---

## Example Queries That Will Show Charts

1. **"At what vehicle age do battery failures occur most frequently?"**
   - Chart: Bar chart (age_bucket vs total_failures)

2. **"Show me failures by month"**
   - Chart: Line chart (month vs failures)

3. **"Top 5 failing parts"**
   - Chart: Bar chart (part vs failure_count)

4. **"Compare failure rates by model"**
   - Chart: Grouped bar chart (model vs failure_rate)

5. **"Failures by city"**
   - Chart: Bar chart (city vs failures)

---

## Benefits

1. **Better UX**: Visual representation is easier to understand than tables
2. **Interactive**: Users can zoom, pan, hover for details
3. **Automatic**: Charts appear for appropriate queries without user asking
4. **No Breaking Changes**: Falls back to table if chart generation fails
5. **Performance**: Charts only generated for chart-friendly queries

---

## Testing Checklist

- [ ] Simple breakdown query ("by X") shows bar chart
- [ ] Time series query ("by month") shows line chart
- [ ] Top N query shows bar chart
- [ ] Large result sets (>20 rows) still show table
- [ ] Single value results don't show chart
- [ ] Chart styling matches app theme (dark mode)
- [ ] Charts are interactive (zoom, pan, hover)
- [ ] Fallback to table if chart generation fails

---

## Future Enhancements

1. **User Preference**: Toggle to show/hide charts
2. **Chart Type Selection**: Let users choose chart type
3. **Export**: Download chart as PNG/PDF
4. **Multiple Charts**: Show multiple charts for complex queries
5. **Chart Annotations**: Add insights/annotations to charts

