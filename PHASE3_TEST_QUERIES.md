# Phase 3 Optimization Test Queries

## Test Suite for Query Result Pagination, Pre-computed Aggregations, and Circuit Breaker

---

## Test 1: Query Result Pagination

Query result pagination automatically adds LIMIT clauses to prevent memory issues with large result sets.

### 1.1 Queries That Should Get LIMIT Added
These queries should automatically get LIMIT 1000 added:

```
Query 1: "Show me all vehicles"
  → Expected: SELECT * FROM historical_data LIMIT 1000
  → Check logs: "Added automatic LIMIT 1000 to prevent large result sets"

Query 2: "List all records"
  → Expected: SELECT * FROM historical_data LIMIT 1000
  → Should have LIMIT added

Query 3: "Find all vehicles with failures"
  → Expected: SELECT * FROM historical_data WHERE failures_count > 0 LIMIT 1000
  → Should have LIMIT added

Query 4: "Show me all VINs"
  → Expected: SELECT vin FROM historical_data LIMIT 1000
  → Should have LIMIT added
```

**Expected**: All should have LIMIT 1000 automatically added (check logs).

---

### 1.2 Queries That Should NOT Get LIMIT Added
These queries should NOT get LIMIT because they return single rows or are already aggregated:

```
Query 1: "Count total failures"
  → Expected: SELECT COUNT(*) FROM historical_data
  → Should NOT have LIMIT (returns single row)

Query 2: "What is the total number of failures?"
  → Expected: SELECT SUM(failures_count) FROM historical_data
  → Should NOT have LIMIT (aggregate, single row)

Query 3: "Show failures by model"
  → Expected: SELECT model, SUM(failures_count) FROM ... GROUP BY model
  → Should NOT have LIMIT (GROUP BY, already aggregated)

Query 4: "Show failures by part"
  → Expected: SELECT primary_failed_part, SUM(...) FROM ... GROUP BY ...
  → Should NOT have LIMIT (GROUP BY, already aggregated)

Query 5: "What is the average failures per vehicle?"
  → Expected: SELECT AVG(failures_count) FROM historical_data
  → Should NOT have LIMIT (aggregate, single row)
```

**Expected**: None should have LIMIT added (check logs).

---

### 1.3 Queries With Existing LIMIT
These queries already have LIMIT, so no additional LIMIT should be added:

```
Query 1: "Show me top 10 failures by model"
  → Expected: SELECT ... GROUP BY model ORDER BY ... LIMIT 10
  → Should keep existing LIMIT 10, not add another

Query 2: "Get first 5 vehicles"
  → Expected: SELECT * FROM historical_data LIMIT 5
  → Should keep existing LIMIT 5
```

**Expected**: Should keep existing LIMIT, not add another.

---

### 1.4 Large Result Set Handling
Test that pagination prevents memory issues:

```
Query: "Show me all data" (if dataset has 50,000+ rows)
  → Expected: Returns first 1000 rows
  → Should complete quickly (not try to load all 50,000 rows)
  → Check memory usage (should be reasonable)
```

**Expected**: Should return quickly with first 1000 rows, not hang or use excessive memory.

---

## Test 2: Pre-computed Aggregations

Pre-computed aggregations provide instant responses for common query patterns.

### 2.1 Queries That Benefit from Pre-computed Tables
These queries should be instant (read from pre-computed tables):

```
Query 1: "Show failures by model"
  → Expected: Reads from precomputed_failures_by_model
  → Should be instant (0-10ms vs 50-200ms for aggregation)

Query 2: "Show failures by part"
  → Expected: Reads from precomputed_failures_by_part
  → Should be instant

Query 3: "Show failures by city"
  → Expected: Reads from precomputed_failures_by_city
  → Should be instant

Query 4: "What is the total number of failures?"
  → Expected: Reads from precomputed_summary
  → Should be instant
```

**How to Verify**:
- Check logs for "Pre-computed aggregations created successfully"
- Compare query times (should be much faster)
- Check SQL query (should query precomputed tables, not historical_data with GROUP BY)

---

### 2.2 Pre-computed Table Structure
Verify pre-computed tables contain correct data:

```
Query 1: "Show failures by model"
  → Check: Results should match manual aggregation
  → Verify: total_failures, record_count, avg_failures_per_record columns

Query 2: "Show failures by part"
  → Check: Results should match manual aggregation
  → Verify: part, total_failures, record_count columns

Query 3: "What is the total number of failures?"
  → Check: Should match SUM(failures_count) from historical_data
  → Verify: total_failures from precomputed_summary
```

**Expected**: Pre-computed results should match manual aggregations exactly.

---

### 2.3 Performance Comparison
Compare performance with and without pre-computed aggregations:

**Without Pre-computation** (if disabled):
- Query time: 50-200ms (full table scan + aggregation)

**With Pre-computation**:
- Query time: 0-10ms (simple table read)

**Test Queries**:
```
Query 1: "Show failures by model"
Query 2: "Show failures by part"
Query 3: "Show failures by city"
```

**Expected**: Pre-computed queries should be 5-20x faster.

---

### 2.4 Pre-computed Table Updates
Test that pre-computed tables are created correctly:

```
Step 1: Check database (if you have access)
  → Verify: precomputed_failures_by_model table exists
  → Verify: precomputed_failures_by_part table exists
  → Verify: precomputed_failures_by_city table exists
  → Verify: precomputed_summary table exists

Step 2: Check indexes
  → Verify: Indexes exist on pre-computed tables
  → Check: idx_precomputed_model, idx_precomputed_part, idx_precomputed_city
```

**Expected**: All pre-computed tables and indexes should exist.

---

## Test 3: Circuit Breaker Pattern

Circuit breaker prevents cascading failures by temporarily stopping requests after too many failures.

### 3.1 Normal Operation (Circuit CLOSED)
Test normal operation when circuit is closed:

```
Query 1: "Show failures by model"
  → Expected: Normal LLM call, succeeds
  → Circuit state: CLOSED
  → Check logs: "Bedrock SQL generation succeeded"

Query 2: "Show failures by part"
  → Expected: Normal LLM call, succeeds
  → Circuit state: CLOSED
```

**Expected**: All queries work normally, circuit stays CLOSED.

---

### 3.2 Circuit Opening (Too Many Failures)
To test circuit opening, you may need to simulate failures:

**Option 1: Rapid Requests (May Cause Throttling)**
```
Step 1: Make 10+ rapid requests in quick succession
  → May trigger throttling errors
Step 2: After 5 failures, circuit should OPEN
  → Check logs: "Circuit breaker: Failure threshold reached (5), opening circuit"
Step 3: Next request should be rejected
  → Check logs: "Circuit breaker: Circuit is OPEN, rejecting request"
  → Expected: Uses pattern-based fallback SQL
```

**Option 2: Simulate Failures (Code Modification)**
Temporarily modify code to simulate failures for testing.

**Expected Behavior**:
- After 5 failures: Circuit opens
- Next requests: Rejected, use fallback
- Logs show: "Circuit breaker: Circuit is OPEN, rejecting request"

---

### 3.3 Circuit Recovery (HALF_OPEN State)
Test automatic recovery after timeout:

```
Step 1: Circuit is OPEN (after 5 failures)
Step 2: Wait 60 seconds (timeout period)
Step 3: Make a request
  → Expected: Circuit moves to HALF_OPEN
  → Check logs: "Circuit breaker: Moving to HALF_OPEN state"
Step 4: If request succeeds 3 times
  → Expected: Circuit moves to CLOSED
  → Check logs: "Circuit breaker: Moving to CLOSED state (recovered)"
```

**Expected**: Circuit automatically recovers after timeout and successful test calls.

---

### 3.4 Fallback Behavior
Test that fallback works when circuit is open:

```
Step 1: Circuit is OPEN
Step 2: Make query: "Show failures by model"
  → Expected: Circuit rejects LLM call
  → Expected: Uses pattern-based SQL generation (fallback)
  → Expected: Query still works (just uses fallback method)
  → Check logs: "Circuit breaker is OPEN - using fallback SQL generation"
```

**Expected**: System continues to work with fallback, graceful degradation.

---

### 3.5 Circuit State Logging
Verify circuit state is logged correctly:

```
Check logs for:
- "Circuit breaker: Moving to HALF_OPEN state"
- "Circuit breaker: Moving to CLOSED state (recovered)"
- "Circuit breaker: Failure threshold reached (5), opening circuit"
- "Circuit breaker: Circuit is OPEN, rejecting request"
```

**Expected**: All state transitions should be logged.

---

## Test 4: Combined Phase 3 Optimizations

Test all Phase 3 optimizations working together.

### 4.1 Full Optimization Flow
```
Query 1: "Show failures by model"
  → Pre-computed aggregation: Instant response (0-10ms)
  → Pagination: Not needed (GROUP BY, already aggregated)
  → Circuit breaker: CLOSED (normal operation)

Query 2: "Show me all vehicles"
  → Pre-computed: Not applicable (not an aggregation query)
  → Pagination: LIMIT 1000 added automatically
  → Circuit breaker: CLOSED (normal operation)

Query 3: "What is the total failures?"
  → Pre-computed: Reads from precomputed_summary (instant)
  → Pagination: Not needed (single row)
  → Circuit breaker: CLOSED (normal operation)
```

**Expected**: All optimizations work together seamlessly.

---

### 4.2 Performance Under Load
Test system behavior under high load:

```
Step 1: Make 20 rapid queries
  → Mix of: "Show failures by model", "Show failures by part", "Show all vehicles"
Step 2: Monitor:
  - Response times (should be fast due to pre-computed aggregations)
  - Memory usage (should be stable due to pagination)
  - Circuit breaker state (should handle failures gracefully)
Step 3: Check for:
  - No memory issues (pagination working)
  - Fast responses (pre-computed aggregations working)
  - Graceful degradation (circuit breaker working)
```

**Expected**: System handles load gracefully with all optimizations.

---

### 4.3 Error Recovery
Test recovery from various error scenarios:

```
Scenario 1: Throttling Errors
  → Circuit breaker opens after 5 failures
  → Uses fallback SQL generation
  → Recovers after 60 seconds

Scenario 2: Network Issues
  → Circuit breaker opens
  → Uses fallback
  → Recovers when network stabilizes

Scenario 3: Service Errors
  → Circuit breaker opens
  → Uses fallback
  → Recovers when service is restored
```

**Expected**: System recovers gracefully from all error scenarios.

---

## Test 5: Edge Cases and Stress Tests

### 5.1 Very Large Datasets
Test pagination with very large datasets:

```
Query: "Show me all data" (if dataset has 100,000+ rows)
  → Expected: Returns first 1000 rows quickly
  → Memory usage: Should be reasonable (not loading all rows)
  → Response time: Should be fast (< 1 second)
```

**Expected**: Pagination prevents memory issues even with very large datasets.

---

### 5.2 Pre-computed Table Accuracy
Verify pre-computed tables stay accurate:

```
Query 1: "Show failures by model" (from pre-computed)
Query 2: Manual query: "SELECT model, SUM(failures_count) FROM historical_data GROUP BY model"
  → Compare results
  → Expected: Should match exactly
```

**Expected**: Pre-computed results should always match manual aggregations.

---

### 5.3 Circuit Breaker State Persistence
Test that circuit breaker state is maintained:

```
Step 1: Make 5 queries that fail (simulate)
  → Circuit opens
Step 2: Make 10 more queries
  → All should use fallback (circuit still open)
Step 3: Wait 60 seconds
Step 4: Make a query
  → Circuit should move to HALF_OPEN
Step 5: Make 3 successful queries
  → Circuit should move to CLOSED
```

**Expected**: Circuit breaker maintains state correctly across multiple requests.

---

### 5.4 Concurrent Queries
Test behavior with concurrent queries:

```
Query 1: "Show failures by model" (starts)
Query 2: "Show failures by part" (starts immediately)
Query 3: "Show all vehicles" (starts immediately)
```

**Expected**: 
- All should benefit from pre-computed aggregations
- Pagination should prevent memory issues
- Circuit breaker should handle failures gracefully

---

## How to Verify Results

### 1. Check Logs for Pagination
Look for:
```
"Added automatic LIMIT 1000 to prevent large result sets"
```

### 2. Check Logs for Pre-computed Aggregations
Look for:
```
"Pre-computed aggregations created successfully"
"Created precomputed_failures_by_model table"
"Created precomputed_failures_by_part table"
```

### 3. Check Logs for Circuit Breaker
Look for:
```
"Circuit breaker: Moving to HALF_OPEN state"
"Circuit breaker: Moving to CLOSED state (recovered)"
"Circuit breaker: Failure threshold reached (5), opening circuit"
"Circuit breaker: Circuit is OPEN, rejecting request"
"Circuit breaker is OPEN - using fallback SQL generation"
```

### 4. Performance Metrics
- **Pagination**: Large queries should return quickly (first 1000 rows)
- **Pre-computed**: Common queries should be instant (0-10ms)
- **Circuit Breaker**: System should recover gracefully from failures

---

## Quick Test Sequence

Run these in order to quickly verify all Phase 3 optimizations:

1. **"Show failures by model"**
   - Pre-computed aggregation: Should be instant
   - Pagination: Not needed (GROUP BY)
   - Circuit breaker: CLOSED (normal)

2. **"Show me all vehicles"**
   - Pre-computed: Not applicable
   - Pagination: LIMIT 1000 added automatically
   - Circuit breaker: CLOSED (normal)

3. **"What is the total failures?"**
   - Pre-computed: Reads from precomputed_summary (instant)
   - Pagination: Not needed (single row)
   - Circuit breaker: CLOSED (normal)

4. **"Count total failures"**
   - Pre-computed: Not applicable (COUNT query)
   - Pagination: Not needed (single row)
   - Circuit breaker: CLOSED (normal)

5. **"Show failures by part"**
   - Pre-computed aggregation: Should be instant
   - Pagination: Not needed (GROUP BY)
   - Circuit breaker: CLOSED (normal)

**Expected Results**:
- Queries 1, 3, 5: Should be instant (pre-computed aggregations)
- Query 2: Should have LIMIT 1000 (pagination)
- Query 4: Should be fast (COUNT, single row)
- All: Should work normally (circuit breaker CLOSED)

---

## Expected Performance Improvements

### Query Result Pagination
- ✅ Prevents memory issues on large datasets
- ✅ Faster initial response (first 1000 rows)
- ✅ Better user experience

### Pre-computed Aggregations
- ✅ Instant responses for common queries (0-10ms vs 50-200ms)
- ✅ Reduced database load
- ✅ Better scalability

### Circuit Breaker
- ✅ Prevents cascading failures
- ✅ Graceful degradation (system continues with fallback)
- ✅ Automatic recovery after failures

---

## Troubleshooting

### If Pagination Not Working
- Check logs for "Added automatic LIMIT" messages
- Verify query doesn't already have LIMIT
- Check that query is not a COUNT or aggregate query

### If Pre-computed Aggregations Not Working
- Check logs for "Pre-computed aggregations created successfully"
- Verify pre-computed tables exist in database
- Compare query times (should be much faster)

### If Circuit Breaker Not Working
- Check logs for circuit breaker state transitions
- Verify failure threshold is reached (5 failures)
- Check timeout period (60 seconds)
- Verify fallback SQL generation is working

---

## Success Criteria

✅ **Query Result Pagination**: Large queries get LIMIT automatically  
✅ **Pre-computed Aggregations**: Common queries are instant (0-10ms)  
✅ **Circuit Breaker**: System recovers gracefully from failures  

All Phase 3 optimizations should work seamlessly together!

---

## Advanced Testing

### Test Pre-computed Table Query Performance
Run these queries and compare execution times:

```
Query 1: "Show failures by model" (should use precomputed_failures_by_model)
  → Expected time: 0-10ms

Query 2: Manual equivalent: "SELECT model, SUM(failures_count) FROM historical_data GROUP BY model"
  → Expected time: 50-200ms (if pre-computed not used)
```

**Expected**: Pre-computed query should be 5-20x faster.

---

### Test Circuit Breaker Recovery Time
Measure recovery time:

```
Step 1: Cause circuit to open (5 failures)
Step 2: Note time
Step 3: Wait for recovery
Step 4: Note time when circuit closes
  → Expected: ~60 seconds (timeout period) + time for 3 successful calls
```

**Expected**: Recovery should happen automatically within timeout period.

---

### Test Memory Usage with Pagination
Monitor memory usage:

```
Query: "Show me all data" (large dataset)
  → Without pagination: High memory usage (loading all rows)
  → With pagination: Low memory usage (only 1000 rows)
```

**Expected**: Pagination should significantly reduce memory usage.

