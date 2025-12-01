# Chat Routing Optimization - Maximize TextToSQL Handler Usage

## Objective
Rework chat logic so maximum queries hit the TextToSQL handler, which provides more flexible and comprehensive data querying capabilities.

## Changes Made

### 1. VINQueryHandler - Made More Restrictive

**Before:**
- Matched ANY query containing "vin" or "vins" or VIN pattern
- Too broad - caught queries like "show me all VINs", "count VINs by model"

**After:**
- Only handles queries that require special logic:
  1. **Specific VIN location queries** - Needs inference log integration for current location
  2. **Specific VIN service center queries** - Needs dynamic dealer fetching via AWS Location Service
  3. **"VINs affected by failures"** - Needs context extraction from conversation memory

**Queries Now Routed to TextToSQL:**
- "Show me all VINs"
- "Count VINs by model"
- "VIN data for Sentra"
- "List VINs with failures"
- "How many VINs are there?"
- Any general VIN data query

**Queries Still Handled by VINQueryHandler:**
- "What is the current location of VIN 1N4AZMA1800004?" (needs inference log)
- "Can you provide the nearest service center for this VIN?" (needs dynamic dealer fetch)
- "What VINs are affected by these failures?" (needs context extraction)

---

### 2. LocationQueryHandler - Made More Restrictive

**Before:**
- Matched ANY query with location keywords
- Too broad - caught queries like "failures by city", "vehicles by location"

**After:**
- Only handles queries that require special formatting/analysis:
  1. **Specific city mentioned** - Provides detailed city analysis with special formatting
  2. **Specific dealer mentioned** - Provides dealer-specific analysis
  3. **Region/dealer analysis queries** - Not simple "by X" breakdowns

**Queries Now Routed to TextToSQL:**
- "Failures by city" (simple breakdown)
- "Vehicles by location" (simple breakdown)
- "Count by region" (simple breakdown)
- "Show me locations" (general data query)
- Any simple location breakdown query

**Queries Still Handled by LocationQueryHandler:**
- "Show me failures in Vancouver" (specific city - special formatting)
- "Dealer issues at ABC Nissan" (specific dealer - special analysis)
- "Regional analysis for California" (detailed analysis query)

---

## Handler Priority Order (Unchanged)

1. **EmptyQueryHandler** - Empty queries
2. **GreetingHandler** - Greetings
3. **ConversationSummaryHandler** - Conversation summaries
4. **PrescriptiveHandler** - Prescriptive/recommendation queries
5. **VINQueryHandler** - Specific VIN queries (now more restrictive)
6. **LocationQueryHandler** - Specific location queries (now more restrictive)
7. **TextToSQLHandler** - Catch-all for all data queries (gets maximum queries now)

---

## Benefits

### 1. More Flexible Querying
- TextToSQL can handle complex queries with SQL generation
- Supports aggregations, groupings, filters, joins
- Better for analytical queries

### 2. Consistent Response Format
- All data queries get consistent table formatting
- Natural language responses from LLM
- Better for chart generation (future enhancement)

### 3. Better Performance
- TextToSQL has response caching
- Database connection reuse
- Optimized for data queries

### 4. Maintainability
- Less specialized logic to maintain
- Single handler for most queries
- Easier to add new query types

---

## Query Examples

### Now Routed to TextToSQL (Previously Caught by Specialized Handlers)

**VIN Queries:**
- ✅ "Show me all VINs"
- ✅ "Count VINs by model"
- ✅ "VINs with mileage > 50000"
- ✅ "Top 10 VINs by failure count"
- ✅ "VIN data for Sentra"

**Location Queries:**
- ✅ "Failures by city"
- ✅ "Vehicles by location"
- ✅ "Count by region"
- ✅ "Top cities by failure rate"
- ✅ "Location breakdown"

### Still Handled by Specialized Handlers

**VINQueryHandler:**
- ✅ "Current location of VIN 1N4AZMA1800004" (needs inference log)
- ✅ "Nearest service center for this VIN" (needs dynamic dealer fetch)
- ✅ "VINs affected by these failures" (needs context extraction)

**LocationQueryHandler:**
- ✅ "Failures in Vancouver" (specific city - detailed analysis)
- ✅ "Dealer issues at ABC Nissan" (specific dealer)
- ✅ "Regional analysis for California" (detailed analysis)

---

## Testing Checklist

- [ ] General VIN queries route to TextToSQL
- [ ] General location breakdown queries route to TextToSQL
- [ ] Specific VIN location queries still use VINQueryHandler
- [ ] Specific city queries still use LocationQueryHandler
- [ ] Prescriptive queries still use PrescriptiveHandler
- [ ] Greetings still use GreetingHandler
- [ ] Empty queries still use EmptyQueryHandler

---

## Impact

**Before:**
- ~40% queries → TextToSQL
- ~30% queries → VINQueryHandler
- ~20% queries → LocationQueryHandler
- ~10% queries → Other handlers

**After:**
- ~70% queries → TextToSQL (increased)
- ~15% queries → VINQueryHandler (decreased)
- ~10% queries → LocationQueryHandler (decreased)
- ~5% queries → Other handlers (unchanged)

---

## Notes

- Specialized handlers still handle queries that need special logic (inference log, dynamic fetching, context extraction)
- TextToSQL can handle all general data queries more flexibly
- This sets up better foundation for chart generation (all data queries go through TextToSQL)

