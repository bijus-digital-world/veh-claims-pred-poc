# Text-to-SQL Architecture for Telematics Analytics

## Overview

This document describes the professional Text-to-SQL implementation for the Telematics Analytics chatbot. This system allows users to ask natural language questions about vehicle data, which are automatically converted to SQL queries and executed.

## Architecture

```
User Question
    ↓
TextToSQLHandler.can_handle() → Check if query is data-focused
    ↓
Schema Generation → Convert pandas DataFrame to SQLite + Generate schema docs
    ↓
LLM (Bedrock) → Convert natural language to SQL query
    ↓
Query Validation → Safety checks, SQL cleaning
    ↓
Query Execution → Execute on SQLite database
    ↓
Result Formatting → Convert results to natural language HTML response
    ↓
User Response
```

## Key Components

### 1. TextToSQLHandler (`chat/handlers_text_to_sql.py`)

**Purpose**: Main handler that orchestrates the Text-to-SQL pipeline.

**Key Methods**:
- `can_handle()`: Determines if query is suitable for Text-to-SQL (data queries vs. specialized analysis)
- `handle()`: Main processing pipeline
- `_prepare_database()`: Converts pandas DataFrame to SQLite
- `_generate_schema_documentation()`: Creates comprehensive schema info for LLM
- `_generate_sql_query()`: Uses LLM to convert natural language to SQL
- `_execute_query_safely()`: Executes SQL with safety checks
- `_generate_natural_language_response()`: Formats results as HTML

### 2. Schema Generation

The system automatically generates schema documentation including:
- Table name and structure
- Column names, data types, and nullability
- Sample values for each column
- Available models, parts, and other domain values
- Common query patterns

**Example Schema Output**:
```
TABLE: historical_data
==================================================

COLUMNS:
  - model (object): 1000/1000 non-null
    Sample values: Sentra, Leaf, Ariya
  - primary_failed_part (object): 950/1000 non-null
    Sample values: Battery, Clutch, Gearbox
  - failures_count (int64): 1000/1000 non-null
    Sample values: 5, 12, 3

BUSINESS LOGIC:
  - Available models: Sentra, Leaf, Ariya, Altima, Rogue
  - Available parts: Battery, Clutch, Gearbox, Brake Caliper
```

### 3. SQL Query Generation

**LLM Prompt Structure**:
1. Database schema information
2. User's natural language question
3. Instructions for SQL generation
4. Safety constraints (SELECT only, no DROP/DELETE)

**Bedrock Integration**:
- Uses AWS Bedrock Claude model (configurable)
- Temperature: 0.1 (low for deterministic SQL)
- Max tokens: 500 (sufficient for complex queries)

**Fallback**: Pattern-based SQL generation for simple queries if Bedrock fails.

### 4. Query Safety

**Security Measures**:
- Only SELECT queries allowed
- Dangerous keywords blocked: DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, TRUNCATE
- SQL injection prevention through parameterized queries
- Query validation before execution

### 5. Result Formatting

Results are formatted based on result size:
- **Single value**: Direct answer display
- **Small results (≤10 rows)**: HTML table
- **Large results (>10 rows)**: Summary + first 10 rows

## Integration with Existing System

### Handler Priority

The `TextToSQLHandler` is placed **early** in the handler chain (after meta queries, before specialized handlers):

```python
1. EmptyQueryHandler
2. GreetingHandler
3. SchemaHandler
4. DateRangeHandler
5. ConversationSummaryHandler
6. **TextToSQLHandler** ← Handles most data queries
7. ModelComparisonHandler
8. ModelRankingHandler
9. ... (other specialized handlers)
10. DefaultHandler
```

### When Text-to-SQL is Used

**Handles**:
- "What's the failure rate for Sentra?"
- "How many failures occurred in 2024?"
- "Show me all Leaf vehicles with battery issues"
- "What's the average mileage for Ariya?"
- "Count failures by model"

**Delegates to Specialized Handlers**:
- "Compare failure rates between models" → ModelComparisonHandler
- "Show me trends" → TrendHandler
- "Prescribe for model Leaf part Battery" → PrescriptiveHandler
- "Top 5 failing parts" → PartRankingHandler

## Advantages Over Intent-Based Handlers

### 1. Flexibility
- Handles new question types without code changes
- Adapts to schema changes automatically
- No need for pattern matching rules

### 2. Maintainability
- Single handler vs. 20+ specialized handlers
- Less code to maintain
- Easier to debug and improve

### 3. Scalability
- Works with any data query
- No need to create handlers for each query pattern
- LLM understands context and relationships

### 4. Better Gen-AI Alignment
- Uses LLM reasoning instead of hardcoded logic
- Natural language understanding
- Context-aware responses

## Example Queries

### Simple Count
**Question**: "How many failures occurred for Sentra?"
**Generated SQL**: `SELECT COUNT(*) FROM historical_data WHERE model = 'Sentra'`
**Response**: "Answer: 245"

### Aggregation
**Question**: "What's the total number of failures by model?"
**Generated SQL**: `SELECT model, SUM(failures_count) as total_failures FROM historical_data GROUP BY model`
**Response**: HTML table with model and total_failures columns

### Filtering
**Question**: "Show me all Leaf vehicles with battery failures"
**Generated SQL**: `SELECT * FROM historical_data WHERE model = 'Leaf' AND primary_failed_part = 'Battery'`
**Response**: HTML table with filtered results

## Configuration

### Bedrock Settings
Configured in `config.py`:
```python
bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
bedrock_max_tokens: int = 500  # Increased for SQL queries
bedrock_temperature: float = 0.1  # Lower for deterministic SQL
```

### Handler Settings
The handler uses:
- SQLite for query execution (temporary database)
- Sample rows: 3 per column for schema generation
- Max result display: 10 rows (with summary for larger results)

## Future Enhancements

1. **Query Caching**: Cache frequently asked queries
2. **Query Optimization**: Analyze and optimize generated SQL
3. **Multi-table Support**: Handle joins across multiple DataFrames
4. **Query Explanation**: Explain what the SQL query does
5. **Error Recovery**: Better handling of SQL errors with suggestions
6. **Evaluation Metrics**: Track SQL generation accuracy
7. **Fine-tuning**: Fine-tune prompts based on query patterns

## Testing

Test queries to validate the system:

1. "What's the failure rate for Sentra?"
2. "How many Leaf vehicles have battery issues?"
3. "Show me all failures where mileage is greater than 50000"
4. "What's the average age of vehicles with clutch failures?"
5. "Count failures by model and part"

## Troubleshooting

### Common Issues

1. **Bedrock API Errors**: Check AWS credentials and region
2. **SQL Generation Fails**: Falls back to pattern-based generation
3. **Query Execution Errors**: Check column names match schema
4. **No Results**: Verify data exists for query conditions

### Debugging

Enable debug logging:
```python
logger.setLevel(logging.DEBUG)
```

Check logs for:
- Generated SQL queries
- Schema information
- Execution errors
- Bedrock API responses

## References

- Based on Text-to-SQL best practices from LangChain
- Adapted from Udemy course architecture
- Enhanced for telematics analytics domain
- Integrated with existing handler system

