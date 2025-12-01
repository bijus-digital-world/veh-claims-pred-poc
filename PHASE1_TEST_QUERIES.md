# Phase 1 Optimization Test Queries

## Test Suite for Database Indexing, LRU Cache, and Query Normalization

---

## Test 1: Database Indexing Performance

These queries should benefit from the indexes we created. Run them and check query execution time in logs.

### 1.1 Model-based queries (uses `idx_model`)
```
Show me failures by model
What is the failure rate for Sentra?
List all Leaf vehicles
Count vehicles by model
```

### 1.2 Part-based queries (uses `idx_primary_failed_part`)
```
Show me failures by part
What parts fail most frequently?
Top 5 failing parts
Count failures for Battery
```

### 1.3 City/Location queries (uses `idx_city`)
```
Show me failures by city
Which city has the most failures?
List failures in Vancouver
```

### 1.4 Dealer queries (uses `idx_dealer_name`)
```
Show me failures by dealer
Which dealers have the most issues?
List failures for United Nissan
```

### 1.5 Composite queries (uses `idx_model_part`)
```
Show me failures by model and part
What are the failure rates for Leaf Battery?
Compare failures for Sentra vs Leaf by part
```

### 1.6 VIN queries (uses `idx_vin`)
```
Show me data for VIN 1N4AZMA1800004
Find all VINs starting with 1N4
Count vehicles by VIN
```

### 1.7 Date-based queries (uses `idx_date`)
```
Show me failures by month
What are the failures by quarter?
List failures by year
```

---

## Test 2: Query Normalization (Cache Hit Rate)

These queries are semantically identical but worded differently. They should hit the same cache entry after normalization.

### 2.1 Synonym variations (should hit same cache)
```
Query 1: Show me failures by model
Query 2: Display failures by model
Query 3: List failures by model
Query 4: Get failures by model
```
**Expected**: All should hit the same cache after the first query.

### 2.2 Singular/Plural variations
```
Query 1: Show me failures by model
Query 2: Show me failure by model
Query 3: Show failures by models
```
**Expected**: Should normalize to same cache key.

### 2.3 Filler word variations
```
Query 1: Show me failures by model
Query 2: Show me the failures by model
Query 3: Show failures by the model
Query 4: Display me failures by model
```
**Expected**: Filler words removed, same cache key.

### 2.4 Case variations
```
Query 1: show me failures by model
Query 2: SHOW ME FAILURES BY MODEL
Query 3: Show Me Failures By Model
```
**Expected**: All normalized to lowercase, same cache key.

---

## Test 3: LRU Cache Behavior

Test that recently used items stay in cache longer.

### 3.1 Sequential access pattern
```
Step 1: Query A - "Show failures by model"
Step 2: Query B - "Show failures by part"
Step 3: Query C - "Show failures by city"
Step 4: Query D - "Show failures by dealer"
... (continue until cache is full, ~100 queries)
Step 101: Query A again - "Show failures by model"
```
**Expected**: Query A should still be cached (LRU keeps recently accessed items).

### 3.2 Repeated access pattern
```
Query 1: "Show failures by model" (cache miss)
Query 2: "Show failures by part" (cache miss)
Query 3: "Show failures by model" (cache hit - should be fast!)
Query 4: "Show failures by part" (cache hit - should be fast!)
Query 5: "Show failures by city" (cache miss)
Query 6: "Show failures by model" (cache hit - still cached!)
```
**Expected**: Repeated queries should hit cache immediately.

---

## Test 4: Combined Optimization Test

Test queries that benefit from all three optimizations.

### 4.1 Indexed + Normalized + Cached
```
Query 1: "Show me failures by model" 
  → Uses idx_model, normalized, cached

Query 2: "Display failures by model"
  → Should hit cache (normalization), no DB query needed

Query 3: "Show failures by part"
  → Uses idx_primary_failed_part, normalized, cached

Query 4: "List failures by part"
  → Should hit cache (normalization), no DB query needed
```

### 4.2 Performance comparison
Run these queries and compare execution times:

**First run** (cold cache, indexes help):
```
"Show me failures by model"
"Show me failures by part"
"Show me failures by city"
```

**Second run** (warm cache, should be instant):
```
"Display failures by model"  (normalized, cached)
"List failures by part"      (normalized, cached)
"Get failures by city"       (normalized, cached)
```

**Expected**: Second run should be significantly faster (cache hits).

---

## Test 5: Edge Cases

### 5.1 Very similar queries (should normalize differently)
```
Query 1: "Show failures by model"
Query 2: "Show failures by models"  (different - "models" vs "model")
```
**Expected**: These might normalize differently - test to see behavior.

### 5.2 Queries with numbers
```
Query 1: "Top 5 failures by model"
Query 2: "Top 10 failures by model"
```
**Expected**: Different cache keys (numbers matter).

### 5.3 Queries with specific values
```
Query 1: "Show failures for Sentra"
Query 2: "Show failures for Leaf"
```
**Expected**: Different cache keys (specific values matter).

---

## How to Verify Results

### 1. Check Logs for Cache Hits
Look for these log messages:
```
"Returning cached response for query (hash: ...)"
"Using cached SQL query (hash: ...)"
```

### 2. Check Logs for Index Usage
SQLite will automatically use indexes when beneficial. Check query execution time in logs.

### 3. Check Logs for Normalization
The query hash is based on normalized query, so similar queries should have similar hash prefixes.

### 4. Performance Metrics
- **First query**: Should see database query execution
- **Cached query**: Should return immediately (no DB query, no LLM call)
- **Normalized query**: Should hit cache even with different wording

---

## Expected Results Summary

### Database Indexing
- ✅ Queries on indexed columns should be 5-10x faster
- ✅ GROUP BY operations should be faster
- ✅ WHERE clauses on indexed columns should be faster

### Query Normalization
- ✅ "Show failures by model" and "Display failures by model" should hit same cache
- ✅ Cache hit rate should increase by 15-25%

### LRU Cache
- ✅ Frequently accessed queries stay in cache longer
- ✅ Cache hit rate should increase by 20-30%
- ✅ Recently used items don't get evicted immediately

---

## Quick Test Sequence

Run these in order to quickly verify all optimizations:

1. **"Show me failures by model"** → First run (cold cache, uses index)
2. **"Display failures by model"** → Should hit cache (normalization works)
3. **"Show failures by part"** → Uses index, cached
4. **"List failures by part"** → Should hit cache (normalization works)
5. **"Show failures by model"** → Should hit cache (LRU keeps it)

All queries after the first should return much faster!

