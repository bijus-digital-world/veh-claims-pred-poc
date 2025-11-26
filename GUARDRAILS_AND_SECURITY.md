# Guardrails and Security for Text-to-SQL Handler

## Overview

This document describes the comprehensive guardrails implemented to prevent hallucinations, ensure data accuracy, and protect PII (Personally Identifiable Information) in the Text-to-SQL handler.

---

## 1. Anti-Hallucination Guardrails

### Prompt Engineering
- **Strict Data-Only Instructions**: LLM is explicitly instructed to use ONLY data from query results
- **Exact Values Section**: Results include an "EXACT VALUES TO USE" section with precise numbers
- **No Estimation Policy**: LLM is forbidden from rounding, approximating, or estimating values
- **Validation Checklist**: LLM must verify every number before including it in response

### Response Validation
- **Number Extraction**: All numbers in LLM response are extracted and validated
- **Value Matching**: Numbers are checked against actual query results with tolerance
- **Hallucination Detection**: Potential mismatches are logged as warnings
- **Debug Mode**: In debug mode, warnings are added to response if hallucinations detected

### Key Features
```python
# Explicit exact values provided to LLM
"EXACT VALUES TO USE (use these exact numbers, do not modify):"
  failures_count = 245
  model = Sentra
  failure_rate = 21.875%
```

---

## 2. PII Protection Guardrails

### Column Filtering
**PII Columns Automatically Filtered:**
- `vin` / `vehicle_identification_number`
- `customer_email` / `email`
- `customer_mobile` / `phone` / `mobile` / `contact`
- `ssn` / `social_security`
- `driver_license` / `passport`
- Any column containing these patterns

### Text Pattern Removal
**PII Patterns Removed from Responses:**
- VIN numbers (17 alphanumeric characters) → `[VIN REDACTED]`
- Email addresses → `[EMAIL REDACTED]`
- Phone numbers (various formats) → `[PHONE REDACTED]`
- SSN patterns → `[SSN REDACTED]`

### Implementation
```python
# PII columns filtered before sending to LLM
safe_results = self._filter_pii_from_dataframe(results.copy())

# PII patterns removed from LLM response
response = self._remove_pii_from_text(response)
```

### Logging
- All PII filtering actions are logged
- Filtered column names are recorded
- PII removal from text is logged

---

## 3. Data Accuracy Validation

### Exact Value Matching
- All numeric values in response are validated against query results
- Tolerance-based matching (0.1% or 0.01 absolute) for rounding differences
- Aggregated values (SUM, AVG, COUNT) are pre-calculated and provided

### Key Aggregated Values
The system automatically calculates and provides:
- Sums of numeric columns
- Averages of numeric columns
- Counts and percentages
- Min/Max values
- Failure rates (if applicable)

### Validation Process
1. Extract all numbers from LLM response
2. Compare against actual query results
3. Check against pre-calculated aggregated values
4. Log warnings for potential mismatches
5. Flag responses that may contain hallucinations

---

## 4. Scope Limitation Guardrails

### Data Scope Restrictions
- **No External Data**: LLM cannot reference data not in query results
- **No Industry Benchmarks**: Comparisons only if benchmark data is in results
- **No Training Data**: LLM cannot use knowledge from training
- **No Assumptions**: Cannot infer or assume missing data

### Prompt Instructions
```
CRITICAL: The "EXACT VALUES TO USE" section above contains the ONLY numbers 
you are allowed to use. Do NOT use any other numbers. Do NOT estimate, round, 
or approximate. Use the exact values provided.
```

---

## 5. Response Filtering

### Multi-Stage Filtering
1. **Pre-LLM**: PII columns filtered from results summary
2. **Post-LLM**: PII patterns removed from response text
3. **Validation**: Numbers validated against actual results
4. **Display**: PII filtered from data tables shown to user

### Safe Data Display
- Data tables only show non-PII columns
- Single value displays filter PII
- All user-facing data is PII-free

---

## 6. Logging and Monitoring

### What's Logged
- **PII Filtering**: Which columns were filtered
- **Hallucination Detection**: Numbers that don't match results
- **Validation Warnings**: Potential accuracy issues
- **Query Execution**: SQL queries and results

### Log Locations
- **Primary**: `logs/chat.log` (all Text-to-SQL activity)
- **Debug Mode**: Additional warnings in response (if enabled)

### Example Log Entries
```
INFO: Filtered PII column: vin
INFO: Filtered PII column: customer_email
WARNING: Potential hallucination detected: Number '12.5%' in response may not match results
```

---

## 7. Configuration

### PII Column List
```python
PII_COLUMNS = [
    'vin', 'vehicle_identification_number',
    'customer_email', 'customer_mobile',
    'customer_phone', 'customer_name',
    'email', 'phone', 'mobile', 'contact',
    'ssn', 'social_security',
    'driver_license', 'passport'
]
```

### Sensitive Business Columns (Optional)
```python
SENSITIVE_BUSINESS_COLUMNS = [
    'supplier_id', 'dealer_id',
    'internal_id', 'employee_id'
]
```

### Tolerance Settings
- **Number Matching**: 0.1% relative or 0.01 absolute tolerance
- **Percentage Matching**: 0.1% tolerance for rounding

---

## 8. Best Practices

### For Developers
1. **Review PII List**: Regularly update PII_COLUMNS based on data schema
2. **Monitor Logs**: Check for hallucination warnings
3. **Test Validation**: Verify number matching works correctly
4. **Update Patterns**: Keep PII regex patterns current

### For Analysts
1. **Verify Key Numbers**: Cross-check important figures
2. **Report Issues**: Flag any suspicious numbers
3. **Use Raw Data**: Check collapsible data tables for verification

---

## 9. Testing Guardrails

### Test Cases

**PII Protection:**
- Query that returns VIN → Should be filtered
- Query that returns email → Should be filtered
- Response mentioning phone → Should be redacted

**Hallucination Prevention:**
- Query returns 21.875% → Response must say exactly 21.875%
- Query returns 245 failures → Response must say exactly 245
- Response should not add "approximately" or "around"

**Data Accuracy:**
- All numbers in response match query results
- Percentages calculated correctly
- Aggregations match SQL results

---

## 10. Limitations and Considerations

### Current Limitations
1. **Pattern-Based PII Detection**: May miss new PII formats
2. **Number Validation**: Basic pattern matching (not semantic)
3. **Tolerance-Based Matching**: May allow small rounding differences

### Future Enhancements
1. **ML-Based PII Detection**: Train model to detect PII patterns
2. **Semantic Validation**: Understand context of numbers
3. **Stricter Validation**: Zero-tolerance for mismatches
4. **Audit Trail**: Track all PII filtering actions

---

## Summary

The Text-to-SQL handler now includes:

✅ **Anti-Hallucination**: Strict prompts, exact values, validation
✅ **PII Protection**: Column filtering, pattern removal, safe display
✅ **Data Accuracy**: Number validation, exact matching, aggregated values
✅ **Scope Limitation**: No external data, no assumptions, results-only
✅ **Multi-Stage Filtering**: Pre-LLM, post-LLM, validation, display
✅ **Comprehensive Logging**: All actions logged for audit

These guardrails ensure that responses are:
- **Accurate**: Only use actual query results
- **Safe**: No PII exposure
- **Reliable**: Validated and verified
- **Professional**: Analyst-friendly formatting

