# Phase 2 Optimization Test Queries

## Test Suite for SQL Result Caching, Retry Logic, and Prompt Optimization

---

## Test 1: SQL Result Caching

SQL result caching stores query results by SQL hash, allowing different queries that generate the same SQL to reuse results.

### 1.1 Same SQL, Different Query Wording
These queries should generate the same SQL and hit the SQL result cache:

```
Query 1: "Show me failures by model"
  → Generates: SELECT model, SUM(failures_count) AS total FROM historical_data GROUP BY model
  → SQL hash: abc123...

Query 2: "Display failures by model"  
  → Generates: SELECT model, SUM(failures_count) AS total FROM historical_data GROUP BY model
  → SQL hash: abc123... (same!)
  → Expected: Should use cached SQL result from Query 1 (instant response)
```

**Test Sequence**:
1. "Show me failures by model" → First run (executes SQL, caches result)
2. "Display failures by model" → Should hit SQL result cache (no DB query)
3. "List failures by model" → Should hit SQL result cache (no DB query)
4. "Get failures by model" → Should hit SQL result cache (no DB query)

**Expected**: Queries 2-4 should return instantly (cached SQL results).

---

### 1.2 Different Queries, Same SQL Pattern
Test queries that generate identical SQL despite different wording:

```
Query 1: "Show failures by part"
Query 2: "Display failures by part"
Query 3: "List failures by part"
Query 4: "Get failures by part"
```

**Expected**: All should generate same SQL and share cached results.

---

### 1.3 SQL Result Cache vs Response Cache
Test the difference between SQL result cache and response cache:

```
Query 1: "Show failures by model"
  → Response cache miss, SQL result cache miss
  → Executes SQL, caches both response and SQL result

Query 2: "Show failures by model" (exact same query)
  → Response cache hit (instant - no LLM, no SQL)

Query 3: "Display failures by model" (different wording, same SQL)
  → Response cache miss (different query hash)
  → SQL result cache hit (same SQL hash)
  → Instant response (no SQL execution, but still needs LLM for response generation)
```

**Expected**: Query 2 is fastest (response cache), Query 3 is fast (SQL result cache).

---

### 1.4 Complex Queries with Same SQL
Test more complex queries that generate identical SQL:

```
Query 1: "Show me failures by city and model"
Query 2: "Display failures grouped by city and model"
Query 3: "List failures by city, model"
```

**Expected**: If they generate the same SQL, they should share cached results.

---

## Test 2: Retry Logic with Exponential Backoff

Test retry behavior on transient failures. Note: You may need to simulate failures to fully test this.

### 2.1 Normal Operation (No Retries)
These should work on first attempt:

```
Query 1: "Show failures by model"
Query 2: "Show failures by part"
Query 3: "Count total failures"
```

**Expected**: All should succeed on first attempt (check logs for "Bedrock SQL generation succeeded").

---

### 2.2 Simulated Throttling (If Possible)
To test retry logic, you may need to:
- Make many rapid requests
- Or temporarily modify code to simulate throttling

**Expected Behavior**:
- First attempt fails with throttling error
- Retries after 1 second
- If still fails, retries after 2 seconds
- If still fails, retries after 4 seconds
- Logs should show: "Bedrock API error (attempt 1/3), retrying in 1s"

---

### 2.3 Error Classification
Test that non-retryable errors don't retry:

```
Query: "Invalid query that causes ValidationException"
```

**Expected**: Should fail immediately without retries (check logs for "Non-retryable Bedrock API error").

---

### 2.4 Network Timeout Recovery
Test recovery from network issues:

```
Query: "Show failures by model" (during network instability)
```

**Expected**: Should retry up to 3 times with exponential backoff.

---

## Test 3: Prompt Optimization

Prompt optimization reduces token usage by including only relevant columns in the schema.

### 3.1 Column-Specific Queries
Test that prompts only include relevant columns:

```
Query 1: "Show failures by model"
  → Expected: Prompt should include: model, failures_count, date (and other common columns)
  → Should NOT include: dealer_name, city, supplier (unless mentioned)

Query 2: "Show failures by city"
  → Expected: Prompt should include: city, failures_count, date
  → Should NOT include: model, dealer_name (unless mentioned)

Query 3: "Show failures by dealer"
  → Expected: Prompt should include: dealer_name, failures_count, date
  → Should NOT include: model, city (unless mentioned)
```

**How to Verify**:
- Check logs for prompt size (should be smaller)
- Compare token usage between queries
- Prompt should be 30-50% smaller than before

---

### 3.2 Multi-Column Queries
Test queries that mention multiple columns:

```
Query: "Show failures by model and part"
  → Expected: Prompt should include: model, primary_failed_part, failures_count, date
  → Should NOT include: city, dealer_name, supplier
```

---

### 3.3 Date-Based Queries
Test date-related queries:

```
Query 1: "Show failures by month"
  → Expected: Prompt should include: date, failures_count
  → Should include date column information

Query 2: "Show failures by quarter"
  → Expected: Prompt should include: date, failures_count
  → Should include date column information
```

---

### 3.4 VIN Queries
Test VIN-specific queries:

```
Query: "Show data for VIN 1N4AZMA1800004"
  → Expected: Prompt should include: vin, and common columns
  → Should NOT include: city, dealer_name (unless mentioned)
```

---

### 3.5 Performance Comparison
Compare LLM call times before and after optimization:

**Before Optimization** (full schema):
- Prompt size: ~800-1000 tokens
- LLM call time: ~500-800ms

**After Optimization** (minimal schema):
- Prompt size: ~300-500 tokens (30-50% reduction)
- LLM call time: ~300-500ms (20-30% faster)

**Test Queries**:
```
Query 1: "Show failures by model"
Query 2: "Show failures by part"
Query 3: "Show failures by city"
```

**Expected**: Each query should have faster LLM response times due to smaller prompts.

---

## Test 4: Combined Phase 2 Optimizations

Test queries that benefit from all Phase 2 optimizations working together.

### 4.1 Full Optimization Flow
```
Query 1: "Show me failures by model"
  → Prompt optimization: Only includes model, failures_count columns
  → LLM generates SQL (faster due to smaller prompt)
  → Executes SQL
  → Caches SQL result
  → Caches response

Query 2: "Display failures by model"
  → Prompt optimization: Only includes model, failures_count columns
  → LLM generates SQL (same SQL as Query 1)
  → SQL result cache hit! (no SQL execution)
  → Generates response from cached SQL result
  → Caches response

Query 3: "Show failures by model"
  → Response cache hit! (instant response)
```

**Expected**: 
- Query 1: Normal speed (all steps)
- Query 2: Fast (SQL result cache hit)
- Query 3: Instant (response cache hit)

---

### 4.2 Error Recovery with Retry
Test retry logic combined with caching:

```
Query 1: "Show failures by model" (first attempt fails with throttling)
  → Retries after 1s
  → Succeeds on retry
  → Caches result

Query 2: "Display failures by model"
  → Should hit SQL result cache (from Query 1)
  → No retry needed (cache hit)
```

**Expected**: Query 2 should be fast even if Query 1 needed retries.

---

## Test 5: Edge Cases and Stress Tests

### 5.1 Cache Eviction
Test LRU cache eviction for SQL results:

```
Step 1: Run 50 different queries (fill SQL result cache)
Step 2: Run Query 1 again: "Show failures by model"
  → Expected: Should still be cached (LRU keeps recently used)
Step 3: Run 10 more new queries
Step 4: Run Query 1 again
  → Expected: May or may not be cached (depends on access pattern)
```

---

### 5.2 Large Result Sets
Test SQL result caching with large results:

```
Query 1: "Show all failures" (returns 10,000+ rows)
  → Should cache result
Query 2: "Display all failures" (same SQL)
  → Should use cached result (instant, no DB query)
```

**Expected**: Large results should still be cached and reused.

---

### 5.3 Empty Results
Test caching behavior with empty results:

```
Query 1: "Show failures for model XYZ" (no results)
  → Should cache empty result
Query 2: "Display failures for model XYZ" (same SQL)
  → Should use cached empty result
```

**Expected**: Empty results should be cached and reused.

---

### 5.4 Concurrent Queries
Test behavior with rapid consecutive queries:

```
Query 1: "Show failures by model" (start)
Query 2: "Show failures by part" (start immediately after)
Query 3: "Show failures by city" (start immediately after)
```

**Expected**: 
- All should benefit from prompt optimization (faster LLM calls)
- If any fail with throttling, retry logic should handle it
- Results should be cached for future use

---

## How to Verify Results

### 1. Check Logs for SQL Result Cache Hits
Look for:
```
"Using cached SQL result (sql_hash: ...)"
"Cached SQL result (sql_hash: ...)"
```

### 2. Check Logs for Retry Attempts
Look for:
```
"Bedrock API error (attempt 1/3), retrying in 1s"
"Bedrock SQL generation succeeded after 2 attempts"
```

### 3. Check Logs for Prompt Optimization
- Compare prompt sizes in logs (should be smaller)
- Check LLM call times (should be faster)
- Verify only relevant columns are included

### 4. Performance Metrics
- **SQL Result Cache Hit**: Should return instantly (no DB query)
- **Retry Success**: Should eventually succeed after retries
- **Prompt Optimization**: 20-30% faster LLM calls, 30-50% fewer tokens

---

## Quick Test Sequence

Run these in order to quickly verify all Phase 2 optimizations:

1. **"Show me failures by model"**
   - First run: Prompt optimization (smaller prompt), LLM call, SQL execution, caches result
   
2. **"Display failures by model"**
   - Should hit SQL result cache (same SQL, instant SQL result)
   - Still needs LLM for response generation (different query wording)
   
3. **"Show failures by model"** (exact same as #1)
   - Should hit response cache (instant - no LLM, no SQL)

4. **"Show failures by part"**
   - Prompt optimization (only part-related columns)
   - New SQL, new cache entry

5. **"List failures by part"**
   - Should hit SQL result cache (same SQL as #4)

**Expected Results**:
- Query 1: Normal speed (all steps)
- Query 2: Fast (SQL result cache hit, but needs LLM for response)
- Query 3: Instant (response cache hit)
- Query 4: Normal speed (new query)
- Query 5: Fast (SQL result cache hit)

---

## Expected Performance Improvements

### SQL Result Caching
- ✅ Different queries with same SQL share cached results
- ✅ Faster responses for semantically similar queries
- ✅ Reduced database load

### Retry Logic
- ✅ Automatic recovery from transient failures
- ✅ 90%+ success rate on throttling errors
- ✅ Better user experience (fewer errors)

### Prompt Optimization
- ✅ 30-50% reduction in prompt tokens
- ✅ 20-30% faster LLM calls
- ✅ Lower API costs
- ✅ Better LLM focus (only relevant columns)

---

## Troubleshooting

### If SQL Result Cache Not Working
- Check logs for "Using cached SQL result" messages
- Verify SQL queries are identical (check SQL hash in logs)
- Ensure cache size hasn't been exceeded

### If Retry Logic Not Working
- Check logs for retry attempts
- Verify error type (should retry on throttling, not validation errors)
- Check retry delay timing (1s, 2s, 4s)

### If Prompt Optimization Not Working
- Check prompt size in logs (should be smaller)
- Verify relevant columns are being extracted
- Check LLM call times (should be faster)

---

## Success Criteria

✅ **SQL Result Caching**: Queries with same SQL hit cache  
✅ **Retry Logic**: Transient failures are automatically retried  
✅ **Prompt Optimization**: Prompts are 30-50% smaller, LLM calls are 20-30% faster  

All Phase 2 optimizations should work seamlessly together!

