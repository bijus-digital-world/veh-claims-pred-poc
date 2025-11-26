# Chatbot Functionality Documentation

This document provides a comprehensive overview of the Vehicle Insights Companion chatbot functionality in the Telematics Analytics application.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [User Interface](#user-interface)
4. [Query Processing Flow](#query-processing-flow)
5. [Handler System](#handler-system)
6. [Intent Classification](#intent-classification)
7. [Conversation Memory](#conversation-memory)
8. [Text-to-SQL Capabilities](#text-to-sql-capabilities)
9. [Voice Input Support](#voice-input-support)
10. [Data Retrieval Methods](#data-retrieval-methods)
11. [Response Generation](#response-generation)
12. [Persistence & State Management](#persistence--state-management)

---

## Overview

The **Vehicle Insights Companion** is an intelligent chatbot integrated into the Telematics Analytics dashboard (Column 3) that allows users to:

- Ask natural language questions about vehicle telematics data
- Get insights on failure rates, trends, and patterns
- Compare models and parts
- Query specific vehicle information
- Generate prescriptive recommendations
- Analyze supplier performance
- Query DTC (Diagnostic Trouble Code) information

**Key Features**:
- Natural language processing
- Context-aware conversations
- Multi-modal input (text and voice)
- Intelligent query routing
- Real-time data analysis
- HTML-formatted responses

**Location**: Dashboard page, Column 3 (below Inference Log Table)

---

## Architecture

### Component Structure

```
┌─────────────────────────────────────────────────────────────┐
│ render_chat_interface() - Main Entry Point (app.py)        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. UI Rendering                                       │  │
│  │    - Chat pane (message history)                     │  │
│  │    - Input form (text input + buttons)               │  │
│  │    - Voice recorder (optional)                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 2. Query Processing                                   │  │
│  │    chat_generate_reply() (chat_helper.py)            │  │
│  │    └─> QueryRouter (chat/handlers.py)                │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 3. Handler Selection                                  │  │
│  │    - Intent Classification                            │  │
│  │    - Handler Routing (30+ handlers)                  │  │
│  │    - Response Generation                              │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 4. Response Display                                   │  │
│  │    - HTML rendering                                   │  │
│  │    - Message history update                           │  │
│  │    - Conversation memory update                       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Key Modules

1. **`app.py`** - Main chat interface rendering (`render_chat_interface()`)
2. **`chat_helper.py`** - Query processing entry point (`generate_reply()`)
3. **`chat/handlers.py`** - Handler base classes and router
4. **`chat/handlers_*.py`** - Specialized handler modules
5. **`chat/intent_classifier.py`** - Intent detection
6. **`chat/conversation_memory.py`** - Conversation context management

---

## User Interface

### Layout

```
┌─────────────────────────────────────────────────────────┐
│ Vehicle Insights Companion                              │
├─────────────────────────────────────────────────────────┤
│ [FAISS Status] [Conversation Stats]                    │
├─────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────┐ │
│ │ [Chat Message Pane - 300px height]                 │ │
│ │                                                     │ │
│ │ User: "What's the failure rate for Sentra?"       │ │
│ │                                                     │ │
│ │ Assistant: "The failure rate for Sentra is..."    │ │
│ │                                                     │ │
│ └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│ [Text Input]  [Send →]  [Clear 🗑]  [🎤 Mic]          │
└─────────────────────────────────────────────────────────┘
```

### Components

1. **Chat Header**:
   - Title: "Vehicle Insights Companion"
   - FAISS availability status
   - Conversation statistics (exchanges, duration)

2. **Message Pane**:
   - Height: 300px
   - Auto-scrolling to latest message
   - User messages: Right-aligned, gray
   - Assistant messages: Left-aligned, blue
   - Timestamps for each message

3. **Input Form**:
   - Text input field
   - Send button (→)
   - Clear chat button (🗑)
   - Voice recording button (🎤) - optional

4. **Styling**:
   - Dark theme matching dashboard
   - Custom CSS for modern appearance
   - Cursor.ai-inspired button design

---

## Query Processing Flow

### Complete Flow Diagram

```
User Input (Text or Voice)
    │
    ├─> [Voice] Transcribe Audio → Text
    │
    ▼
Input Validation
    │
    ├─> Empty? → EmptyQueryHandler
    │
    ▼
Intent Classification
    │
    ├─> empty → EmptyQueryHandler
    ├─> small_talk/off_domain/safety → GenericIntentHandler
    └─> data_request → Continue
    │
    ▼
Handler Router
    │
    ├─> Check each handler in order
    ├─> First handler that can_handle() = True
    └─> Execute handler.handle()
    │
    ▼
Response Generation
    │
    ├─> Query data (SQL, pandas, FAISS/TF-IDF)
    ├─> Process results
    └─> Format as HTML
    │
    ▼
Display & Persist
    │
    ├─> Add to chat_history
    ├─> Update conversation_memory
    └─> Save to disk (CSV)
```

### Step-by-Step Process

1. **User Input Reception**
   - Text input via form submission
   - Voice input via audio transcription (AWS Transcribe)

2. **Input Validation**
   - Check for empty/whitespace-only queries
   - Check for duplicate submissions

3. **Intent Classification**
   - Analyze query text for domain keywords
   - Classify as: `empty`, `data_request`, `small_talk`, `off_domain`, `safety`

4. **Handler Selection**
   - Router iterates through handlers in priority order
   - Each handler's `can_handle()` method is called
   - First handler returning `True` processes the query

5. **Query Execution**
   - Handler accesses historical data (`df_history`)
   - May use SQL, pandas operations, or vector search
   - Retrieves relevant records/data

6. **Response Formatting**
   - Data formatted into human-readable text
   - HTML markup applied for styling
   - Tables, lists, or paragraphs as appropriate

7. **State Updates**
   - Message added to `st.session_state.chat_history`
   - Exchange added to `conversation_memory`
   - Chat history persisted to CSV file

---

## Handler System

### Handler Architecture

**Base Class**: `QueryHandler` (ABC)

All handlers inherit from `QueryHandler` and implement:
- `can_handle(context: QueryContext) -> bool`: Determines if handler can process query
- `handle(context: QueryContext) -> str`: Processes query and returns HTML response

### Handler Categories

#### 1. Meta Handlers (Processed First)

| Handler | Purpose | Example Query |
|---------|---------|---------------|
| `EmptyQueryHandler` | Handle empty queries | "" |
| `GreetingHandler` | Handle greetings | "Hello", "Hi" |
| `SchemaHandler` | List dataset columns | "What columns are available?" |
| `DateRangeHandler` | Show data time span | "How many months of data?" |

#### 2. Context Handlers

| Handler | Purpose | Example Query |
|---------|---------|---------------|
| `ConversationSummaryHandler` | Show conversation stats | "What have we discussed?" |
| `ContextAwareHandler` | Use conversation context | Follow-up questions |

#### 3. Text-to-SQL Handler

| Handler | Purpose | Example Query |
|---------|---------|---------------|
| `TextToSQLHandler` | Convert NL to SQL queries | "Show me all vehicles with mileage > 50000" |

#### 4. Comparison Handlers

| Handler | Purpose | Example Query |
|---------|---------|---------------|
| `ModelComparisonHandler` | Compare models | "Compare Leaf vs Ariya failure rates" |

#### 5. Ranking Handlers

| Handler | Purpose | Example Query |
|---------|---------|---------------|
| `ModelRankingHandler` | Rank models by metric | "Which models have the most failures?" |
| `PartRankingHandler` | Rank parts by metric | "Top 5 failing parts" |

#### 6. Model-Specific Handlers

| Handler | Purpose | Example Query |
|---------|---------|---------------|
| `ModelSpecificRateHandler` | Failure rate for specific model | "Failure rate for Sentra" |
| `ModelSpecificCountHandler` | Failure count for specific model | "How many failures for Leaf?" |
| `ModelSpecificDTCHandler` | DTC codes for specific model | "DTC codes for Ariya" |

#### 7. Supplier Handlers

| Handler | Purpose | Example Query |
|---------|---------|---------------|
| `SupplierListHandler` | List suppliers | "Show all suppliers" |
| `DefectiveSupplierHandler` | Identify problematic suppliers | "Which suppliers have high failure rates?" |

#### 8. Location & VIN Handlers

| Handler | Purpose | Example Query |
|---------|---------|---------------|
| `VINQueryHandler` | Query by VIN | "Details for VIN 1N4AAPA1902317" |
| `LocationQueryHandler` | Query by location | "Failures in Vancouver" |
| `FailureReasonHandler` | Common failure reasons | "What are the common failure reasons?" |

#### 9. DTC Handlers

| Handler | Purpose | Example Query |
|---------|---------|---------------|
| `DTCCommonCodesHandler` | Most common DTC codes | "Most common DTC codes" |
| `DTCByModelHandler` | DTC codes by model | "DTC codes for Leaf" |
| `DTCByManufacturingYearHandler` | DTC by year | "DTC codes from 2020" |
| `DTCTrendHandler` | DTC trends over time | "DTC trend analysis" |

#### 10. Distribution Handlers

| Handler | Purpose | Example Query |
|---------|---------|---------------|
| `MileageDistributionHandler` | Mileage distribution | "Mileage distribution" |
| `AgeDistributionHandler` | Age distribution | "Age distribution of vehicles" |

#### 11. Analysis Handlers

| Handler | Purpose | Example Query |
|---------|---------|---------------|
| `PrescriptiveHandler` | Generate recommendations | "Prescribe for Leaf Battery" |
| `TimeToResolutionHandler` | Time to repair analysis | "Average time to resolution" |
| `TotalMetricHandler` | Total metrics | "Total failures" |
| `CountAndAverageHandler` | Count and average | "Count and average failures" |
| `MonthlyAggregateHandler` | Monthly aggregations | "Monthly failure trends" |
| `RateHandler` | Failure rates | "Overall failure rate" |
| `TrendHandler` | Trend analysis | "Failure trends over time" |
| `TopFailedPartsHandler` | Top failing parts | "Top 10 failed parts" |
| `IncidentDetailsHandler` | Incident details | "Details of incident X" |

#### 12. Fallback Handlers

| Handler | Purpose | Example Query |
|---------|---------|---------------|
| `ContextAwareHandler` | Context-aware responses | Follow-ups using conversation |
| `DefaultHandler` | Catch-all handler | Any query not matched above |

### Handler Priority Order

Handlers are checked in this order (first match wins):

```
1. EmptyQueryHandler
2. GreetingHandler
3. SchemaHandler
4. DateRangeHandler
5. ConversationSummaryHandler
6. TextToSQLHandler
7. ModelComparisonHandler
8. ModelRankingHandler
9. PartRankingHandler
10. ModelSpecificRateHandler
11. ModelSpecificCountHandler
12. SupplierListHandler
13. DefectiveSupplierHandler
14. VINQueryHandler
15. FailureReasonHandler
16. LocationQueryHandler
17. ModelSpecificDTCHandler
18. DTCCommonCodesHandler
19. DTCByModelHandler
20. DTCByManufacturingYearHandler
21. DTCTrendHandler
22. MileageDistributionHandler
23. AgeDistributionHandler
24. PrescriptiveHandler
25. TimeToResolutionHandler
26. TotalMetricHandler
27. CountAndAverageHandler
28. MonthlyAggregateHandler
29. RateHandler
30. TrendHandler
31. TopFailedPartsHandler
32. IncidentDetailsHandler
33. ContextAwareHandler
34. DefaultHandler (catch-all)
```

---

## Intent Classification

### Intent Labels

The `classify_intent()` function in `chat/intent_classifier.py` classifies queries into:

1. **`empty`**: Empty or whitespace-only queries
2. **`data_request`**: Queries requesting vehicle/telematics data
3. **`small_talk`**: Casual conversation (greetings, chit-chat)
4. **`off_domain`**: Queries unrelated to vehicle analytics
5. **`safety`**: Safety-related queries (filtered for appropriate responses)

### Classification Logic

```python
def classify_intent(query: str) -> IntentResult:
    # Check for empty
    if not query.strip():
        return "empty"
    
    # Check for safety patterns
    if safety_patterns_detected:
        return "safety"
    
    # Check for small talk patterns
    if small_talk_patterns_detected:
        return "small_talk"
    
    # Check for domain keywords
    domain_keywords = ["failure", "vehicle", "model", "part", "battery", ...]
    domain_hits = count_matching_keywords(query, domain_keywords)
    
    if domain_hits >= 1 and has_question_word:
        return "data_request"
    
    if domain_hits >= 2:
        return "data_request"
    
    return "off_domain"
```

### Domain Keywords

Keywords that indicate telematics/vehicle domain:
- Vehicle terms: `vehicle`, `car`, `fleet`, `model`, `part`
- Failure terms: `failure`, `failures`, `failed`, `claim`, `claims`
- Technical terms: `battery`, `voltage`, `soc`, `temperature`, `dtc`
- Analysis terms: `trend`, `rate`, `count`, `ranking`
- Metadata: `column`, `field`, `schema`, `data`

---

## Conversation Memory

### ConversationContext Class

The `ConversationContext` class (`chat/conversation_memory.py`) manages conversation state:

**Features**:
- Maintains conversation history
- Tracks recent exchanges (configurable window)
- Provides context for follow-up questions
- Supports conversation summarization

**Configuration**:
- `context_window`: Number of recent exchanges to keep in active context (default: 10)
- `max_memory_size`: Maximum total exchanges to keep (default: 100)

### ChatExchange Structure

Each exchange stores:
```python
{
    "query": str,                    # User's question
    "response": str,                 # Assistant's response (HTML)
    "timestamp": str,                # ISO timestamp
    "exchange_id": str,              # Unique ID
    "handler_used": Optional[str],   # Which handler processed it
    "processing_time_ms": Optional[float],  # Response time
    "context_used": Optional[Dict]   # Additional context
}
```

### Context Usage

**Recent Context**:
- Last N exchanges available for context-aware responses
- Used by `ContextAwareHandler` for follow-up questions

**Related Exchanges**:
- Finds exchanges related to current query by keyword matching
- Helps provide continuity in conversations

**Example**:
```
User: "What's the failure rate for Sentra?"
Assistant: "The failure rate for Sentra is 3.2%..."

User: "How about Leaf?"  [Follow-up]
Assistant: "The failure rate for Leaf is 2.8%..."  [Uses context from previous exchange]
```

---

## Text-to-SQL Capabilities

### TextToSQLHandler

The `TextToSQLHandler` (`chat/handlers_text_to_sql.py`) converts natural language queries into SQL operations.

**Process**:

1. **Database Preparation**:
   - Creates temporary SQLite database from `df_history`
   - Normalizes column names (spaces → underscores, lowercase)
   - Handles NA/null values

2. **Schema Documentation**:
   - Generates schema description with column types
   - Includes sample values for each column
   - Provides context for LLM SQL generation

3. **SQL Generation**:
   - Uses LLM (Bedrock) to generate SQL from natural language
   - Includes safety checks (PII column filtering)
   - Validates SQL syntax

4. **Query Execution**:
   - Executes SQL safely (read-only operations)
   - Handles errors gracefully
   - Returns results as pandas DataFrame

5. **Response Generation**:
   - Converts SQL results to natural language
   - Formats data as HTML tables or summaries

**Example**:
```
User: "Show me all Sentra vehicles with mileage greater than 50000"

Generated SQL:
SELECT * FROM historical_data 
WHERE model = 'Sentra' 
AND mileage > 50000 
LIMIT 100

Response: "Found 45 Sentra vehicles with mileage > 50,000 miles..."
```

**Safety Features**:
- PII column filtering (VIN, email, phone numbers)
- Read-only operations (no DELETE, UPDATE, DROP)
- Row limit enforcement (default: 100)
- SQL injection protection

---

## Voice Input Support

### Voice Recording Flow

1. **Audio Capture**:
   - User clicks microphone button
   - Browser records audio (using `streamlit-audiorecorder`)
   - Audio stored as WAV format

2. **Audio Processing**:
   - Audio optimized (16kHz mono for faster processing)
   - Stored in session state (`pending_audio_bytes`)
   - Unique audio ID generated for deduplication

3. **Transcription**:
   - Audio sent to AWS Transcribe
   - Language: Configurable (default: English)
   - Returns transcribed text

4. **Query Processing**:
   - Transcribed text processed as regular text query
   - Same handler routing applies
   - Response generated normally

### Voice Configuration

**Requirements**:
- AWS credentials configured
- IAM permissions for Transcribe service
- `streamlit-audiorecorder` package installed
- FFmpeg installed (for audio processing)

**Configuration Options**:
- `VOICE_ENABLED`: Enable/disable voice feature (environment variable)
- `TRANSCRIBE_LANGUAGE_CODE`: Language for transcription (default: "en-US")
- `S3_BUCKET`: S3 bucket for audio storage (optional)

---

## Data Retrieval Methods

### 1. Direct DataFrame Operations

**When Used**: Simple aggregations, filtering, grouping

**Example Handlers**: `ModelSpecificRateHandler`, `TotalMetricHandler`

**Process**:
```python
# Filter by model
df_filtered = df_history[df_history['model'] == 'Sentra']

# Calculate failure rate
failures = df_filtered['failures_count'].sum()
total = len(df_filtered)
rate = (failures / total) * 100 if total > 0 else 0
```

### 2. SQL Queries (Text-to-SQL)

**When Used**: Complex queries, user-specified filters, aggregations

**Example Handler**: `TextToSQLHandler`

**Process**:
- Convert NL to SQL
- Execute on SQLite database
- Return results

### 3. Vector Search (FAISS/TF-IDF)

**When Used**: Semantic similarity, finding related records

**Example Usage**: Finding similar incidents, related vehicle records

**Methods**:

**FAISS** (Preferred if available):
- Pre-computed embeddings from historical data
- Fast similarity search
- Loaded from disk at startup

**TF-IDF** (Fallback):
- Term frequency-inverse document frequency
- Built on-the-fly from historical data
- Cached in session state

**Process**:
```python
# Retrieve similar records
results = retrieve_with_faiss_or_tfidf(
    query="battery failure",
    faiss_res=faiss_res,
    tfidf_vect=VECT_CHAT,
    tfidf_X=X_CHAT,
    tfidf_rows=HISTORY_ROWS_CHAT,
    top_k=5
)
```

### 4. LLM-Based Retrieval

**When Used**: Complex analysis, prescriptive summaries

**Example Handler**: `PrescriptiveHandler`

**Process**:
- Query sent to AWS Bedrock (LLM)
- LLM analyzes data and generates response
- Includes historical context and trends

---

## Response Generation

### Response Format

All responses are **HTML-formatted strings** containing:

1. **Structured Content**:
   - Paragraphs (`<p>`)
   - Lists (`<ul>`, `<ol>`)
   - Tables (`<table>`)
   - Bold/emphasis (`<strong>`, `<em>`)

2. **Styling**:
   - Inline CSS for consistent appearance
   - Color coding for metrics (red/green/yellow)
   - Tables with borders and spacing

3. **Data Presentation**:
   - Numbers formatted with commas
   - Percentages with % symbol
   - Dates in readable format

### Example Response Structure

```html
<p><strong>Failure Rate Analysis for Sentra</strong></p>
<p>The failure rate for Sentra is <strong style="color:#fca5a5;">3.2%</strong>, 
   based on <strong>1,234</strong> vehicles in the dataset.</p>
<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th>Part</th>
    <th>Failures</th>
    <th>Rate</th>
  </tr>
  <tr>
    <td>Engine Cooling System</td>
    <td>45</td>
    <td>3.6%</td>
  </tr>
</table>
```

### Response Types

1. **Simple Text**: Single paragraph answers
2. **Metric Values**: Numeric results with formatting
3. **Tables**: Structured data in HTML tables
4. **Lists**: Ranked or bulleted lists
5. **Charts**: (Future enhancement) Visual data representation
6. **Recommendations**: Prescriptive actions with explanations

---

## Persistence & State Management

### Chat History

**Storage**: `st.session_state.chat_history`

**Structure**:
```python
[
    {
        "role": "user",  # or "assistant"
        "text": "...",   # Query or response (HTML for assistant)
        "ts": "2025-01-XX...",  # ISO timestamp
        "id": "uuid..."  # Unique message ID
    },
    ...
]
```

**Persistence**:
- Saved to CSV file: `logs/chat_log.csv`
- Persisted after each exchange
- Best-effort (errors logged but don't block)

### Conversation Memory

**Storage**: `st.session_state.conversation_memory`

**Type**: `ConversationContext` object

**Persistence**:
- Saved to session state as dictionary
- Loaded on startup if available
- Maintains conversation continuity across sessions

### TF-IDF Index

**Storage**: `st.session_state` (cached)

**Keys**:
- `chat_tfidf_built`: Boolean flag
- `VECT_CHAT`: Vectorizer object
- `X_CHAT`: TF-IDF matrix
- `HISTORY_ROWS_CHAT`: Row metadata

**Initialization**:
- Built once on first chat interaction
- Cached for subsequent queries
- Rebuilt if data changes

### FAISS Index

**Storage**: Disk files (loaded at startup)

**Files**:
- `data/vectors/historical_data_index.faiss`: FAISS index
- `data/vectors/historical_data_embs.npy`: Embeddings
- `data/vectors/historical_data_meta.npy`: Metadata

**Loading**:
- Loaded via `load_persisted_faiss()` (cached)
- Available if files exist and FAISS library installed
- Falls back to TF-IDF if unavailable

---

## Error Handling

### Error Types & Responses

1. **Empty Query**:
   - Prompt user to ask a question
   - No error shown, just helpful message

2. **Handler Failure**:
   - Error logged
   - Generic error message shown to user
   - Query processing continues with next handler

3. **Data Access Error**:
   - Error message with helpful hints
   - Suggests checking column names or data availability

4. **SQL Generation Error**:
   - Shows generated SQL snippet (for debugging)
   - Provides tips for reformulating query

5. **Transcription Error**:
   - Error message about AWS setup
   - Suggests checking credentials/permissions

### Logging

All errors are logged with:
- Error message
- Stack trace (if debug mode)
- Handler name (if applicable)
- Query text (truncated)

**Log Levels**:
- `DEBUG`: Detailed processing information
- `INFO`: Handler routing, successful operations
- `WARNING`: Non-critical issues (fallbacks)
- `ERROR`: Failures that affect functionality

---

## Configuration

### Environment Variables

- `VOICE_ENABLED`: Enable voice input (true/false)
- `AWS_REGION`: AWS region for Transcribe/Bedrock
- `S3_BUCKET`: S3 bucket for audio storage

### Configuration File (`config.py`)

- `config.model.voice_enabled`: Voice feature flag
- `config.model.transcribe_language_code`: Transcription language
- `config.paths.chat_log_file`: Chat history file path
- `config.paths.faiss_index_path`: FAISS index file path
- `config.debug`: Enable debug logging

---

## Example Queries

### Failure Analysis
- "What's the failure rate for Sentra?"
- "Show me failure trends over time"
- "Which parts fail most often?"

### Comparisons
- "Compare Leaf vs Ariya failure rates"
- "Which model has the most failures?"

### Rankings
- "Top 5 failing parts"
- "Which models have the highest failure rates?"

### Specific Queries
- "Details for VIN 1N4AAPA1902317"
- "Failures in Vancouver"
- "DTC codes for Leaf"

### Prescriptive
- "Prescribe for Leaf Battery"
- "What should I do about high failure rates?"

### Data Exploration
- "Show me all vehicles with mileage > 50000"
- "What columns are available?"
- "How many months of data do we have?"

---

## Performance Considerations

### Optimization Strategies

1. **Caching**:
   - TF-IDF index cached in session state
   - FAISS index loaded once at startup
   - Historical data cached (`@st.cache_data`)

2. **Lazy Loading**:
   - Handlers loaded only when needed
   - Database created only for Text-to-SQL queries

3. **Response Time**:
   - Simple queries: < 500ms
   - Complex SQL queries: 1-3 seconds
   - LLM-based responses: 3-10 seconds

4. **Memory Management**:
   - Chat history limited (persisted to disk)
   - Conversation memory window limited (10 exchanges)
   - Temporary databases cleaned up after use

---

## Future Enhancements

### Planned Features

1. **Visual Charts**: Graph generation for trends
2. **Export Functionality**: Download chat history
3. **Multi-language Support**: Translation capabilities
4. **Custom Handlers**: Plugin system for custom queries
5. **Enhanced Context**: Better follow-up understanding
6. **Streaming Responses**: Real-time response streaming

### Known Limitations

1. **Voice Input**: Requires AWS setup
2. **Complex Queries**: May need rephrasing
3. **Data Volume**: Large datasets may slow responses
4. **PII Handling**: Some personal info may appear in responses

---

## Troubleshooting

### Common Issues

1. **No Response Generated**:
   - Check handler routing logs
   - Verify data availability
   - Try rephrasing query

2. **Voice Not Working**:
   - Check AWS credentials
   - Verify `streamlit-audiorecorder` installed
   - Check FFmpeg installation

3. **Slow Responses**:
   - Check data size
   - Verify FAISS index loaded
   - Check network connectivity (for LLM)

4. **Incorrect Responses**:
   - Check intent classification
   - Verify handler selection
   - Review data quality

---

## Version History

- **2025-01-XX**: Initial documentation
- Based on implementation in `app.py`, `chat_helper.py`, and `chat/` modules

---

## Additional Resources

- **Handler Documentation**: See individual handler files in `chat/handlers_*.py`
- **Text-to-SQL Architecture**: `chat/TEXT_TO_SQL_ARCHITECTURE.md`
- **Test Questions**: `chat/TEXT_TO_SQL_TEST_QUESTIONS.md`
- **Complex Test Cases**: `chat/COMPLEX_TEST_QUESTIONS.md`
- **Security Guide**: `chat/GUARDRAILS_AND_SECURITY.md`

