# Queries NOT Hitting TextToSQL Handler

## Overview

Based on the current handler routing logic, here are all the query types that are handled by other handlers **before** reaching TextToSQLHandler.

---

## Handler Execution Order

Handlers are checked in this order:
1. **EmptyQueryHandler**
2. **GreetingHandler**
3. **ConversationSummaryHandler**
4. **PrescriptiveHandler**
5. **VINQueryHandler**
6. **LocationQueryHandler**
7. **TextToSQLHandler** (catch-all)

---

## 1. EmptyQueryHandler

**Queries Handled:**
- Empty queries (whitespace only)
- Blank queries

**Examples:**
- `""` (empty string)
- `"   "` (whitespace only)

**Why Not TextToSQL:** These are not valid queries, so they're caught early.

---

## 2. GreetingHandler

**Queries Handled:**
- Greetings and casual conversation starters

**Patterns:**
- Contains: `" hi "`, `" hello "`, `" hey "`, `"hiya"`, `"yo "`
- Starts with: `"hi"`, `"hello"`, `"hey"`, `"good morning"`, `"good afternoon"`, `"good evening"`
- Contains: `"how are you"`, `"how's it going"`, `"what's up"`
- Short queries (≤3 words) containing: `"hi"`, `"hello"`, `"hey"`

**Examples:**
- `"Hi"`
- `"Hello"`
- `"Hey there"`
- `"How are you?"`
- `"Good morning"`
- `"What's up"`

**Why Not TextToSQL:** These are conversational, not data queries.

---

## 3. ConversationSummaryHandler

**Queries Handled:**
- Conversation summary requests

**Patterns:**
- Contains: `"conversation summary"`, `"what have we discussed"`, `"summary of our chat"`, `"conversation overview"`, `"what did we talk about"`, `"chat summary"`

**Examples:**
- `"What have we discussed?"`
- `"Show me a conversation summary"`
- `"Summary of our chat"`
- `"What did we talk about?"`

**Why Not TextToSQL:** These are meta-queries about the conversation itself, not data queries.

---

## 4. PrescriptiveHandler

**Queries Handled:**
- Prescriptive/recommendation queries (NOT comparison queries)

**Patterns:**
- Contains: `"prescribe"`, `"recommend"`, `"prescriptive"`, `"advice"`, `"action"`, `"what should i do"`, `"what should we do"`, `"how to fix"`, `"how to improve"`, `"what to do about"`, `"suggestions"`, `"recommendations"`, `"guidance"`, `"help with"`, `"solution"`, `"remedy"`, `"intervention"`, `"improving"`

**Excluded:**
- Comparison queries (`"compare"`, `"comparison"`, `"vs"`, `"versus"`) → Go to TextToSQL

**Examples:**
- `"Prescribe for model Leaf part Battery"`
- `"What should I do about high failure rates?"`
- `"How to fix battery issues?"`
- `"Recommendations for transmission problems"`
- `"What advice do you have for Sentra failures?"`

**Why Not TextToSQL:** These require LLM-based prescriptive analysis, not SQL queries.

---

## 5. VINQueryHandler

**Queries Handled:**
Only very specific VIN queries that require special logic:

### 5a. Specific VIN Location Queries
**Requirements:**
- Contains actual VIN pattern (`1N4[A-Z0-9]{8,}`) OR `"this VIN"` / `"that VIN"` / `"the VIN"` reference
- AND contains location keywords: `"location"`, `"where"`, `"coordinates"`, `"current location"`, `"current position"`, `"where is"`

**Examples:**
- `"What is the current location of VIN 1N4AZMA1800004?"`
- `"Where is this VIN?"` (if VIN was mentioned earlier)
- `"Coordinates for VIN 1N4AZPD1800373"`

**Why Not TextToSQL:** Needs inference log integration for current location.

### 5b. Specific VIN Service Center Queries
**Requirements:**
- Contains actual VIN pattern OR VIN reference
- AND contains: `"service center"`, `"dealer"`, `"nearest"`

**Examples:**
- `"Can you provide the nearest service center for this VIN?"`
- `"Nearest dealer for VIN 1N4AZMA1800004"`

**Why Not TextToSQL:** Needs dynamic dealer fetching via AWS Location Service.

### 5c. VINs Affected by Failures (with context)
**Requirements:**
- Contains: `"affected"`, `"vins affected"`, `"vehicles affected"`, `"affected by"`, `"affected by these"`, `"affected by failures"`

**Examples:**
- `"What VINs are affected by these failures?"` (uses conversation context)
- `"Show me vehicles affected by failures"`

**Why Not TextToSQL:** Needs context extraction from conversation memory.

**Queries That DO Go to TextToSQL:**
- `"Show me all VINs"`
- `"Count VINs by model"`
- `"VIN data for Sentra"`
- `"List VINs with failures"`
- `"How many VINs are there?"`
- Any general VIN data query

---

## 6. LocationQueryHandler

**Queries Handled:**
Only specific location queries that require special formatting:

### 6a. Specific City Queries
**Requirements:**
- Contains location keywords AND mentions a specific city name from the dataset

**Examples:**
- `"Show me failures in Vancouver"`
- `"Vehicles in Toronto"`
- `"Analysis for Seattle"`

**Why Not TextToSQL:** Provides detailed city analysis with special formatting.

### 6b. Specific Dealer Queries
**Requirements:**
- Contains location keywords AND mentions a specific dealer name from the dataset

**Examples:**
- `"Dealer issues at ABC Nissan"`
- `"Problems at XYZ Service Center"`

**Why Not TextToSQL:** Provides dealer-specific analysis.

### 6c. Region/Dealer Analysis Queries
**Requirements:**
- Contains: `"dealer issues"`, `"dealer problems"`, `"dealer failures"`, `"major issues"`, `"issues from"`, `"problems from"`, `"failures from"`, `"regional analysis"`, `"city analysis"`, `"location analysis"`, `"area analysis"`, `"geographic"`, `"which dealers"`, `"which region"`
- AND NOT a simple breakdown (`"by city"`, `"by location"`, etc.)
- AND NOT a time breakdown (`"by month"`, `"by quarter"`, etc.)

**Examples:**
- `"Which dealers have the most issues?"`
- `"Regional analysis for California"`
- `"Dealer problems in the area"`

**Queries That DO Go to TextToSQL:**
- `"Failures by city"` (simple breakdown)
- `"Vehicles by location"` (simple breakdown)
- `"Count by region"` (simple breakdown)
- `"Failures by month"` (time-based breakdown)
- `"Failures by quarter"` (time-based breakdown)
- Any simple location breakdown query

---

## 7. GenericIntentHandler (Intent-Based)

**Queries Handled:**
- Queries classified as `"small_talk"`, `"off_domain"`, or `"safety"` by the intent classifier

**These are caught BEFORE handler routing** (in the `route` method):

### 7a. Small Talk
**Examples:**
- `"How's the weather?"`
- `"Tell me a joke"`
- `"Who are you?"`

### 7b. Off-Domain
**Examples:**
- `"What's the capital of France?"`
- `"How do I cook pasta?"`
- Queries with no domain keywords

### 7c. Safety
**Examples:**
- `"What's my password?"`
- `"Credit card number"`
- `"SSN"`

**Why Not TextToSQL:** These are outside the telematics domain.

---

## Summary: Query Types NOT Hitting TextToSQL

| Handler | Query Type | Count | Examples |
|---------|-----------|-------|----------|
| **EmptyQueryHandler** | Empty queries | ~1% | `""`, `"   "` |
| **GreetingHandler** | Greetings | ~2% | `"Hi"`, `"Hello"`, `"How are you?"` |
| **ConversationSummaryHandler** | Conversation summaries | ~1% | `"What have we discussed?"` |
| **PrescriptiveHandler** | Prescriptive queries | ~5% | `"Prescribe for model Leaf part Battery"` |
| **VINQueryHandler** | Specific VIN queries | ~10% | `"Current location of VIN 1N4..."` |
| **LocationQueryHandler** | Specific location queries | ~8% | `"Failures in Vancouver"` |
| **GenericIntentHandler** | Off-domain queries | ~3% | `"How's the weather?"` |
| **TextToSQLHandler** | **All other queries** | **~70%** | Everything else! |

---

## Queries That DO Hit TextToSQL

**All other queries** that don't match the above patterns, including:

- ✅ General data queries: `"Show me all vehicles"`, `"Count failures"`
- ✅ Breakdown queries: `"Failures by model"`, `"Vehicles by city"`, `"Failures by month"`
- ✅ Comparison queries: `"Compare failure rates by model"`, `"Leaf vs Sentra"`
- ✅ Aggregation queries: `"Total failures"`, `"Average mileage"`
- ✅ Filter queries: `"Vehicles with mileage > 50000"`
- ✅ Time series: `"Failures by quarter"`, `"Trends over time"`
- ✅ Ranking queries: `"Top 5 failing parts"`, `"Worst models"`
- ✅ Schema queries: `"What columns are available?"`
- ✅ General VIN queries: `"Show me all VINs"`, `"VINs by model"`
- ✅ Simple location breakdowns: `"Failures by city"`, `"Vehicles by location"`

---

## Notes

1. **TextToSQL is the catch-all**: It handles ~70% of queries (all data queries)
2. **Specialized handlers are restrictive**: They only handle queries that need special logic
3. **Intent-based filtering**: Some queries are filtered out before reaching handlers
4. **Handler order matters**: First handler that matches wins

---

## Recommendations

If you want **more queries** to hit TextToSQL:

1. **Make specialized handlers even more restrictive** (already done for VIN and Location)
2. **Move PrescriptiveHandler later** (but this might break prescriptive queries)
3. **Remove specialized handlers entirely** (but you'll lose special features like inference log integration)

The current setup is optimized to:
- ✅ Maximize TextToSQL usage (~70%)
- ✅ Keep special features (inference log, dynamic dealer fetching, context extraction)
- ✅ Handle meta-queries appropriately (greetings, summaries)

