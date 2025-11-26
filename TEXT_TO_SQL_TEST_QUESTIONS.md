# Text-to-SQL Test Questions for Telematics Analytics

## Recommended Test Questions

This document provides a comprehensive list of test questions organized by category to validate the Text-to-SQL handler functionality.

---

## Category 1: Simple Count & Aggregation Queries

**Purpose**: Test basic SQL generation and single-value responses

1. **"What's the failure rate for Sentra?"**
   - Expected: Count or percentage calculation
   - Tests: Model filtering, aggregation

2. **"How many total failures occurred?"**
   - Expected: SUM of failures_count
   - Tests: Simple aggregation

3. **"What's the total number of records in the dataset?"**
   - Expected: COUNT(*)
   - Tests: Basic counting

4. **"How many Leaf vehicles are in the data?"**
   - Expected: COUNT with model filter
   - Tests: Filtering by model

5. **"What's the average mileage across all vehicles?"**
   - Expected: AVG(mileage)
   - Tests: Average calculation

---

## Category 2: Filtering & Conditional Queries

**Purpose**: Test WHERE clause generation and filtering logic

6. **"Show me all failures where battery voltage is less than 300"**
   - Expected: SELECT with WHERE battery_voltage < 300
   - Tests: Numeric filtering

7. **"Find all Sentra vehicles with clutch failures"**
   - Expected: Filter by model AND primary_failed_part
   - Tests: Multiple conditions

8. **"What failures occurred in 2024?"**
   - Expected: Date filtering
   - Tests: Date/time handling

9. **"Show me vehicles with mileage greater than 50000"**
   - Expected: WHERE mileage > 50000
   - Tests: Comparison operators

10. **"List all failures for Leaf model where age is less than 2 years"**
    - Expected: Multiple filters combined
    - Tests: Complex WHERE clauses

---

## Category 3: Group By & Aggregation Queries

**Purpose**: Test GROUP BY and aggregate functions

11. **"What's the failure rate by model?"**
    - Expected: GROUP BY model with COUNT/SUM
    - Tests: Grouping and aggregation

12. **"Show me total failures grouped by part"**
    - Expected: GROUP BY primary_failed_part
    - Tests: Grouping by part

13. **"What's the average mileage by model?"**
    - Expected: GROUP BY model, AVG(mileage)
    - Tests: Grouping with averages

14. **"Count failures by model and part"**
    - Expected: GROUP BY model, primary_failed_part
    - Tests: Multiple grouping columns

15. **"What's the maximum and minimum mileage by model?"**
    - Expected: GROUP BY with MAX and MIN
    - Tests: Multiple aggregate functions

---

## Category 4: Comparison & Ranking Queries

**Purpose**: Test comparison logic and ordering

16. **"Which model has the most failures?"**
    - Expected: GROUP BY with ORDER BY DESC LIMIT 1
    - Tests: Ranking queries

17. **"Compare failure rates between Sentra and Leaf"**
    - Expected: Filter for both models, compare
    - Tests: Comparison queries (may delegate to ModelComparisonHandler)

18. **"What are the top 5 failing parts?"**
    - Expected: GROUP BY part, ORDER BY DESC, LIMIT 5
    - Tests: Top N queries (may delegate to PartRankingHandler)

19. **"Which model has the highest average mileage?"**
    - Expected: GROUP BY with AVG, ORDER BY DESC
    - Tests: Aggregation with ordering

20. **"Show me models sorted by failure count"**
    - Expected: GROUP BY model, ORDER BY count
    - Tests: Sorting results

---

## Category 5: Complex Multi-Condition Queries

**Purpose**: Test complex SQL with multiple conditions

21. **"Find all Sentra vehicles with battery failures where mileage is between 30000 and 50000"**
    - Expected: Multiple WHERE conditions with BETWEEN
    - Tests: Complex filtering

22. **"Show me failures for Leaf or Ariya models with critical health status"**
    - Expected: IN clause or OR conditions
    - Tests: Multiple value matching

23. **"What's the failure count for vehicles manufactured in 2022 with battery issues?"**
    - Expected: Multiple filters including date
    - Tests: Date and part filtering

24. **"List all vehicles where battery_voltage < 300 OR battery_temperature > 60"**
    - Expected: OR conditions
    - Tests: OR logic

25. **"Find failures where model is Sentra AND (part is Battery OR part is Clutch)"**
    - Expected: AND/OR combination
    - Tests: Complex boolean logic

---

## Category 6: Time-Based & Trend Queries

**Purpose**: Test date/time handling (may delegate to specialized handlers)

26. **"What failures occurred in the last 6 months?"**
    - Expected: Date filtering with relative dates
    - Tests: Date calculations

27. **"Show me failures by month"**
    - Expected: GROUP BY with date extraction
    - Tests: Time-based grouping

28. **"What's the failure trend over time?"**
    - Expected: Time series grouping
    - Tests: Trend analysis (may delegate to TrendHandler)

29. **"How many failures occurred in each quarter?"**
    - Expected: GROUP BY quarter
    - Tests: Quarterly aggregation

---

## Category 7: Location & Geographic Queries

**Purpose**: Test location-based filtering

30. **"Show me failures in California"**
    - Expected: Filter by city/state/region
    - Tests: Location filtering

31. **"What's the failure count by city?"**
    - Expected: GROUP BY city
    - Tests: Geographic grouping

32. **"Find failures near latitude 40.0 and longitude -75.0"**
    - Expected: Geographic proximity filtering
    - Tests: Coordinate-based queries

---

## Category 8: Component & Sensor Data Queries

**Purpose**: Test telematics sensor data queries

33. **"What's the average battery voltage for Leaf vehicles?"**
    - Expected: AVG with model filter
    - Tests: Sensor data aggregation

34. **"Show me vehicles with battery SOC below 20%"**
    - Expected: Filter by battery_soc
    - Tests: Sensor threshold filtering

35. **"What's the maximum coolant temperature recorded?"**
    - Expected: MAX(coolant_temperature)
    - Tests: Sensor data max/min

36. **"List vehicles where brake pressure is outside normal range (25-35 bar)"**
    - Expected: Range filtering
    - Tests: Range conditions

37. **"What's the average engine RPM by model?"**
    - Expected: GROUP BY model, AVG(engine_rpm)
    - Tests: Sensor data grouping

---

## Category 9: Supplier & Quality Queries

**Purpose**: Test supplier-related queries

38. **"Which supplier has the most defective parts?"**
    - Expected: GROUP BY supplier, ORDER BY defect count
    - Tests: Supplier analysis

39. **"Show me failures by supplier quality score"**
    - Expected: GROUP BY supplier_quality_score
    - Tests: Quality-based grouping

40. **"What's the average defect rate by supplier?"**
    - Expected: GROUP BY supplier, AVG(defect_rate)
    - Tests: Supplier metrics

---

## Category 10: VIN & Specific Vehicle Queries

**Purpose**: Test specific vehicle lookups

41. **"Show me all data for VIN 3N1Z5FMF..."**
    - Expected: Filter by VIN
    - Tests: Exact match filtering

42. **"What's the failure history for a specific vehicle?"**
    - Expected: VIN-based query
    - Tests: Vehicle-specific queries

---

## Category 11: Health Status & Diagnostic Queries

**Purpose**: Test health status and DTC queries

43. **"How many vehicles have critical health status?"**
    - Expected: COUNT with health_status filter
    - Tests: Status filtering

44. **"Show me all vehicles with battery health status as Warning"**
    - Expected: Filter by battery_health_status
    - Tests: Component health filtering

45. **"What DTC codes are most common?"**
    - Expected: GROUP BY dtc_code
    - Tests: DTC analysis (may delegate to DTC handlers)

---

## Category 12: Edge Cases & Error Testing

**Purpose**: Test error handling and edge cases

46. **"What's the failure rate for a model that doesn't exist?"**
    - Expected: Empty results with appropriate message
    - Tests: No results handling

47. **"Show me all data"**
    - Expected: SELECT * with LIMIT
    - Tests: Broad queries

48. **"What columns are available?"**
    - Expected: Schema information (may delegate to SchemaHandler)
    - Tests: Meta queries

---

## Recommended Testing Sequence

### Phase 1: Basic Functionality (Start Here)
1. Question #1: "What's the failure rate for Sentra?"
2. Question #2: "How many total failures occurred?"
3. Question #6: "Show me all failures where battery voltage is less than 300"

### Phase 2: Aggregation & Grouping
11. Question #11: "What's the failure rate by model?"
12. Question #12: "Show me total failures grouped by part"
13. Question #13: "What's the average mileage by model?"

### Phase 3: Complex Queries
21. Question #21: "Find all Sentra vehicles with battery failures where mileage is between 30000 and 50000"
22. Question #22: "Show me failures for Leaf or Ariya models with critical health status"

### Phase 4: Natural Language Quality
Test questions that should produce excellent analyst-friendly responses:
- Question #1: Should provide context about failure rates
- Question #11: Should compare models and provide insights
- Question #16: Should highlight the top model and explain significance

---

## Expected Behaviors

### Text-to-SQL Handler Should Handle:
✅ Simple counts and aggregations
✅ Filtering queries
✅ Group by queries
✅ Basic comparisons
✅ Sensor data queries
✅ Location queries

### Should Delegate to Specialized Handlers:
🔄 Trend analysis → TrendHandler
🔄 Model comparisons → ModelComparisonHandler
🔄 Rankings → ModelRankingHandler / PartRankingHandler
🔄 Prescriptive analysis → PrescriptiveHandler
🔄 DTC-specific queries → DTC handlers

---

## Success Criteria

For each question, verify:
1. ✅ SQL query is generated correctly
2. ✅ Query executes without errors
3. ✅ Results are accurate
4. ✅ Natural language response is professional and analyst-friendly
5. ✅ Response includes context and insights
6. ✅ HTML formatting is clean and readable
7. ✅ Raw data is accessible (if applicable)

---

## Logging to Monitor

Check `logs/chat.log` for:
- "TextToSQLHandler processing query"
- Generated SQL queries (DEBUG level)
- "Natural language response generated successfully"
- Any errors or fallbacks

---

## Notes

- Some questions may be handled by specialized handlers (comparisons, trends, rankings)
- This is expected behavior - Text-to-SQL handles data queries, specialized handlers handle analysis
- The system should gracefully delegate when appropriate
- Natural language responses should be professional and domain-specific

