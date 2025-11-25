"""
Text-to-SQL Handler for Telematics Analytics

Converts natural language queries into SQL operations using LLM.
"""

import re
import html as _html
import pandas as pd
import sqlite3
from typing import Optional, Tuple, List
import json

from chat.handlers import QueryHandler, QueryContext
from utils.logger import chat_logger as logger
from config import config


class TextToSQLHandler(QueryHandler):
    
    DATA_QUERY_KEYWORDS = [
        'what', 'how many', 'count', 'total', 'sum', 'average', 'avg', 'mean',
        'show', 'list', 'find', 'get', 'display', 'which', 'where', 'when',
        'filter', 'group by', 'aggregate', 'maximum', 'minimum', 'max', 'min'
    ]
    
    SPECIALIZED_KEYWORDS = [
        'prescribe', 'recommend', 'trend', 'forecast', 'predict', 'pattern',
        'compare', 'ranking', 'top', 'worst', 'best'
    ]
    
    PII_COLUMNS = [
        'vin', 'vehicle_identification_number', 'customer_email', 'customer_mobile',
        'customer_phone', 'customer_name', 'email', 'phone', 'mobile', 'contact',
        'ssn', 'social_security', 'driver_license', 'passport'
    ]
    
    def __init__(self):
        self.sqlite_conn = None
        
    def can_handle(self, context: QueryContext) -> bool:
        query_lower = context.query_lower
        
        # catch "show all data" type queries
        has_show_all_data = bool(re.search(r'\b(show|display|list|get|find)\s+(me\s+)?all\s+data\b', query_lower))
        if has_show_all_data:
            return True
        
        # distinguish "top 5 vehicles" (individual records) from "top 5 models" (aggregated)
        # need to allow words between number and vehicles like "top 10 Sentra vehicles"
        has_top_for_records = bool(re.search(r'\btop\s+\d+.*?\b(vehicles?|records?|rows?|entries?|items?)\b', query_lower))
        
        # handle individual record queries even if "top" is in specialized keywords
        if has_top_for_records:
            # look for domain terms
            has_data_patterns = bool(re.search(r'\b(model|part|vin|mileage|age|failure|failures|claim|claims|repair|repairs|battery|batteries|supplier|suppliers|dtc|component|components|issue|issues|fault|faults|health|soc|temperature|dealer|dealers|clutch|brake|engine|transmission|gearbox)\b', query_lower))
            # check for model names too
            has_model_name = bool(re.search(r'\b(sentra|leaf|ariya|altima|rogue|pathfinder|frontier|titan)\b', query_lower))
            if has_data_patterns or has_model_name:
                return True
        
        # skip if specialized keywords found - let other handlers deal with it
        special_hits = [keyword for keyword in self.SPECIALIZED_KEYWORDS if keyword in query_lower]
        if special_hits:
            allow_due_to_dtc = "dtc" in query_lower and any(kw in special_hits for kw in ["top", "ranking", "rank"])
            if not allow_due_to_dtc:
                return False
        
        has_data_keywords = any(keyword in query_lower for keyword in self.DATA_QUERY_KEYWORDS)
        has_data_patterns = bool(re.search(r'\b(model|part|vin|mileage|age|failure|failures|claim|claims|repair|repairs|battery|batteries|supplier|suppliers|dtc|component|components|issue|issues|fault|faults|health|soc|temperature|dealer|dealers|clutch|brake|engine|transmission|gearbox)\b', query_lower))
        has_model_name = bool(re.search(r'\b(sentra|leaf|ariya|altima|rogue|pathfinder|frontier|titan)\b', query_lower))
        
        if has_data_patterns or has_model_name:
            return True
        
        # need domain terms for generic question words
        domain_terms = [
            'vehicle', 'car', 'fleet', 'failure', 'claim', 'repair', 'battery',
            'sensor', 'telematics', 'supplier', 'dealer', 'dtc', 'fault', 'issue',
            'warranty', 'model', 'part', 'mileage', 'age', 'health', 'soc', 'voltage', 'data'
        ]
        has_domain_term = any(term in query_lower for term in domain_terms)
        return has_data_keywords and has_domain_term
    
    def handle(self, context: QueryContext) -> str:
        logger.info("TextToSQLHandler processing query")
        
        try:
            schema_info, sqlite_db_path = self._prepare_database(context.df_history)
            sql_query = self._generate_sql_query(context.query, schema_info, context)
            
            if not sql_query:
                return "<p>I couldn't generate a valid SQL query for your question. Please try rephrasing.</p>"
            
            results = self._execute_query_safely(sql_query, sqlite_db_path, context.query)
            
            if results is None:
                # include SQL snippet for debugging
                sql_snippet = sql_query[:150] if sql_query else "N/A"
                return (f"<p>I couldn't execute the query. Please check if your question references valid columns and values.</p>"
                        f"<p style='font-size: 0.85em; color: #94a3b8; margin-top: 8px;'>"
                        f"<strong>Generated SQL:</strong> <code>{_html.escape(sql_snippet)}...</code></p>"
                        f"<p style='font-size: 0.85em; color: #94a3b8;'>"
                        f"Tip: For date-based queries (quarters, months), ensure a date column exists in the dataset.</p>")
            
            response = self._generate_natural_language_response(
                context.query, 
                sql_query, 
                results,
                context
            )
            
            return response
            
        except Exception as e:
            logger.error(f"TextToSQLHandler failed: {e}", exc_info=True)
            return f"<p>Error processing your query: {_html.escape(str(e))}</p>"
        finally:
            self._cleanup_database()
    
    def _prepare_database(self, df: pd.DataFrame) -> Tuple[str, str]:
        import tempfile
        import os
        
        temp_dir = tempfile.gettempdir()
        db_path = os.path.join(temp_dir, f"telematics_temp_{pd.Timestamp.now().value}.db")
        
        conn = sqlite3.connect(db_path)
        
        df_clean = df.copy()
        na_regex = re.compile(r'^\s*(?:n/?a|na|none|null|not\s+available|not\s+applicable|-)\s*$', re.IGNORECASE)
        object_columns = df_clean.select_dtypes(include=["object"]).columns
        for col in object_columns:
            df_clean[col] = df_clean[col].apply(
                lambda value: pd.NA if isinstance(value, str) and na_regex.match(value) else value
            )
        df_clean.columns = [col.replace(' ', '_').replace('-', '_').lower() for col in df_clean.columns]
        
        df_clean.to_sql('historical_data', conn, if_exists='replace', index=False)
        
        schema_info = self._generate_schema_documentation(df_clean, conn)
        
        self.sqlite_conn = conn
        return schema_info, db_path
    
    def _generate_schema_documentation(self, df: pd.DataFrame, conn: sqlite3.Connection) -> str:
        schema_parts = []
        
        schema_parts.append("TABLE: historical_data")
        schema_parts.append("=" * 50)
        
        schema_parts.append("\nCOLUMNS:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null_count = df[col].notna().sum()
            total_count = len(df)
            
            sample_values = df[col].dropna().head(3).tolist()
            sample_str = ", ".join([str(v)[:30] for v in sample_values if pd.notna(v)])
            
            schema_parts.append(f"  - {col} ({dtype}): {non_null_count}/{total_count} non-null")
            if sample_str:
                schema_parts.append(f"    Sample values: {sample_str}")
        
        schema_parts.append("\nDATE COLUMNS:")
        date_columns = []
        for col in df.columns:
            try:
                if 'date' in col.lower():
                    date_columns.append(col)
                    continue
                if df[col].dtype in ['datetime64[ns]']:
                    date_columns.append(col)
                    continue
                if df[col].dtype == 'object':
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    if parsed.notna().any():
                        date_columns.append(col)
            except Exception:
                continue
        if date_columns:
            for col in date_columns[:3]:  # Show first 3 date columns
                sample_dates = df[col].dropna().head(2).tolist()
                schema_parts.append(f"  - {col}: Use for date/time operations (sample: {', '.join([str(d)[:20] for d in sample_dates])})")
        else:
            schema_parts.append("  - No date columns found")
        
        schema_parts.append("\nBUSINESS LOGIC:")
        if 'model' in df.columns:
            models = df['model'].dropna().unique()[:5]
            schema_parts.append(f"  - Available models: {', '.join([str(m) for m in models])}")
        
        if 'primary_failed_part' in df.columns:
            parts = df['primary_failed_part'].dropna().unique()[:5]
            schema_parts.append(f"  - Available parts: {', '.join([str(p) for p in parts])}")
        
        schema_parts.append("\nCOMMON QUERIES:")
        schema_parts.append("  - Count records: SELECT COUNT(*) FROM historical_data")
        schema_parts.append("  - Filter by model: SELECT * FROM historical_data WHERE model = 'Sentra'")
        schema_parts.append("  - Group by: SELECT model, COUNT(*) FROM historical_data GROUP BY model")
        if date_columns:
            date_col = date_columns[0]
            schema_parts.append(f"  - Group by quarter: SELECT CASE WHEN strftime('%m', {date_col}) IN ('01','02','03') THEN 'Q1' WHEN strftime('%m', {date_col}) IN ('04','05','06') THEN 'Q2' WHEN strftime('%m', {date_col}) IN ('07','08','09') THEN 'Q3' ELSE 'Q4' END AS quarter, COUNT(*) FROM historical_data GROUP BY quarter")
        
        return "\n".join(schema_parts)
    
    def _generate_sql_query(self, user_query: str, schema_info: str, context: QueryContext) -> Optional[str]:
        try:
            prompt = self._build_sql_generation_prompt(user_query, schema_info)
            
            try:
                sql_query = self._call_bedrock_for_sql(prompt, context)
                if not sql_query:
                    logger.info("Bedrock failed, using pattern-based SQL generation")
                    sql_query = self._pattern_based_sql_generation(user_query, schema_info)
            except Exception as e:
                logger.warning(f"Bedrock call failed: {e}, using pattern-based fallback")
                sql_query = self._pattern_based_sql_generation(user_query, schema_info)
            
            sql_query = self._clean_sql_query(sql_query)
            
            return sql_query
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}", exc_info=True)
            return None
    
    def _build_sql_generation_prompt(self, user_query: str, schema_info: str) -> str:
        prompt = f"""You are a SQL expert. Convert the following natural language question into a SQL query.

DATABASE SCHEMA:
{schema_info}

USER QUESTION: {user_query}

INSTRUCTIONS:
1. Generate ONLY a valid SQLite SQL query
2. Do NOT include any explanations or markdown formatting
3. Use proper column names from the schema
4. Handle NULL values appropriately
5. Use appropriate aggregations (COUNT, SUM, AVG) when needed
6. Return the SQL query in a single line if possible

DATE/TIME HANDLING (SQLite):
- For quarters: Use CASE WHEN strftime('%m', date_column) IN ('01','02','03') THEN 'Q1' WHEN strftime('%m', date_column) IN ('04','05','06') THEN 'Q2' WHEN strftime('%m', date_column) IN ('07','08','09') THEN 'Q3' ELSE 'Q4' END AS quarter
- For months: Use strftime('%Y-%m', date_column) AS month
- For years: Use strftime('%Y', date_column) AS year
- For year-month: Use strftime('%Y-%m', date_column) AS year_month
- Always check if date column exists in schema before using date functions
- If date column is named 'date', use it. If named differently (e.g., 'manufacturing_date', 'incident_date'), use the actual column name from schema

VIN/PARTIAL MATCHING:
- When user provides a partial VIN (contains "..." or is shorter than 17 characters), use LIKE with wildcard
- Examples:
  * "VIN 3N1Z5FMF..." → SELECT * FROM historical_data WHERE vin LIKE '3N1Z5FMF%'
  * "VIN starting with 1N4" → SELECT * FROM historical_data WHERE vin LIKE '1N4%'
  * "VIN 1N4AAMF1100965" (complete) → SELECT * FROM historical_data WHERE vin = '1N4AAMF1100965'
- For partial text matches in other columns, also use LIKE when appropriate

GROUPING/BREAKDOWN QUERIES:
- When user asks "by X", "per X", or "grouped by X", you MUST use GROUP BY to show breakdown
- DO NOT return aggregated totals - return the breakdown for each group
- Examples:
  * "failure count by city" → SELECT city, SUM(failures_count) AS total_failures FROM historical_data GROUP BY city
  * "failures by model" → SELECT model, SUM(failures_count) AS total_failures FROM historical_data GROUP BY model
  * "failures by part" → SELECT primary_failed_part, SUM(failures_count) AS total_failures FROM historical_data GROUP BY primary_failed_part

EXAMPLES:
- "How many failures by quarter?" → SELECT CASE WHEN strftime('%m', date) IN ('01','02','03') THEN 'Q1' WHEN strftime('%m', date) IN ('04','05','06') THEN 'Q2' WHEN strftime('%m', date) IN ('07','08','09') THEN 'Q3' ELSE 'Q4' END AS quarter, SUM(failures_count) AS total_failures FROM historical_data GROUP BY quarter
- "Failures by month" → SELECT strftime('%Y-%m', date) AS month, SUM(failures_count) AS total_failures FROM historical_data GROUP BY month
- "Failures by year" → SELECT strftime('%Y', date) AS year, SUM(failures_count) AS total_failures FROM historical_data GROUP BY year
- "Failure count by city" → SELECT city, SUM(failures_count) AS total_failures FROM historical_data GROUP BY city
- "Show me all data for VIN 3N1Z5FMF..." → SELECT * FROM historical_data WHERE vin LIKE '3N1Z5FMF%'
- "VIN 1N4AAMF1100965" (complete VIN) → SELECT * FROM historical_data WHERE vin = '1N4AAMF1100965'

SQL QUERY:"""
        
        return prompt
    
    def _call_bedrock_for_sql(self, prompt: str, context: QueryContext) -> Optional[str]:
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
            model_id = config.model.bedrock_model_id
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            }
            
            response = bedrock.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            
            response_body = json.loads(response['body'].read())
            sql_query = response_body['content'][0]['text'].strip()
            
            return sql_query
            
        except ClientError as e:
            logger.error(f"Bedrock API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Bedrock SQL generation failed: {e}", exc_info=True)
            return None
    
    def _pattern_based_sql_generation(self, user_query: str, schema_info: str) -> Optional[str]:
        query_lower = user_query.lower()
        
        # Check for VIN queries
        if 'vin' in query_lower:
            # Match VIN pattern: alphanumeric, possibly with ... at the end
            vin_match = re.search(r'vin\s+([A-Z0-9]+(?:\.\.\.)?)', user_query, re.IGNORECASE)
            if vin_match:
                vin_value = vin_match.group(1).strip()
                vin_clean = vin_value.replace('...', '').replace('.', '').strip()
                # Check if partial VIN (contains ... or is shorter than typical VIN length ~17 chars)
                if '...' in vin_value or len(vin_clean) < 10:
                    return f"SELECT * FROM historical_data WHERE vin LIKE '{vin_clean}%'"
                else:
                    return f"SELECT * FROM historical_data WHERE vin = '{vin_clean}'"
        
        # Check for date-based grouping (quarter, month, year)
        if 'quarter' in query_lower or 'q1' in query_lower or 'q2' in query_lower:
            # Try to find date column from schema
            date_col = 'date'  # Default
            if 'date' in schema_info.lower():
                # Try to extract date column name from schema
                date_match = re.search(r'(\w*date\w*)', schema_info, re.IGNORECASE)
                if date_match:
                    date_col = date_match.group(1)
            
            if 'failure' in query_lower:
                metric = 'SUM(failures_count)'
            else:
                metric = 'COUNT(*)'
            
            return f"SELECT CASE WHEN strftime('%m', {date_col}) IN ('01','02','03') THEN 'Q1' WHEN strftime('%m', {date_col}) IN ('04','05','06') THEN 'Q2' WHEN strftime('%m', {date_col}) IN ('07','08','09') THEN 'Q3' ELSE 'Q4' END AS quarter, {metric} AS total FROM historical_data WHERE {date_col} IS NOT NULL GROUP BY quarter"
        
        if 'count' in query_lower or 'how many' in query_lower:
            if 'model' in query_lower:
                model_match = re.search(r"model\s+(\w+)", query_lower)
                if model_match:
                    model = model_match.group(1)
                    return f"SELECT COUNT(*) FROM historical_data WHERE model = '{model}'"
            return "SELECT COUNT(*) FROM historical_data"
        
        elif 'total' in query_lower or 'sum' in query_lower:
            if 'failure' in query_lower:
                return "SELECT SUM(failures_count) FROM historical_data"
            return "SELECT COUNT(*) FROM historical_data"
        
        elif 'average' in query_lower or 'avg' in query_lower:
            if 'mileage' in query_lower:
                return "SELECT AVG(mileage) FROM historical_data"
            return "SELECT COUNT(*) FROM historical_data"
        
        return "SELECT * FROM historical_data LIMIT 10"
    
    def _clean_sql_query(self, sql_query: str) -> str:
        if not sql_query:
            return ""
        
        sql_query = re.sub(r'```sql\s*', '', sql_query)
        sql_query = re.sub(r'```\s*', '', sql_query)
        sql_query = sql_query.strip()
        sql_query = re.sub(r'^(SELECT|select)\s+', 'SELECT ', sql_query, flags=re.IGNORECASE)
        
        # convert partial VIN = to LIKE
        vin_equals_pattern = r"WHERE\s+vin\s*=\s*'([A-Z0-9]+\.?\.?\.?)'"
        def replace_vin_match(match):
            vin_value = match.group(1)
            vin_clean = vin_value.replace('...', '').replace('.', '').strip()
            # partial VIN gets LIKE
            if '...' in vin_value or len(vin_clean) < 10:
                return f"WHERE vin LIKE '{vin_clean}%'"
            return match.group(0)  # exact match for full VINs
        
        sql_query = re.sub(vin_equals_pattern, replace_vin_match, sql_query, flags=re.IGNORECASE)
        
        # fix LIKE patterns that still have ... in them
        vin_like_pattern = r"vin\s+LIKE\s+'([A-Z0-9]+)\.\.\.'"
        sql_query = re.sub(vin_like_pattern, r"vin LIKE '\1%'", sql_query, flags=re.IGNORECASE)
        
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        for keyword in dangerous_keywords:
            if keyword in sql_query.upper():
                logger.warning(f"Potentially dangerous SQL keyword detected: {keyword}")
                return ""
        
        sql_lower = sql_query.lower()
        for non_null_column in ['dtc_code']:
            if non_null_column in sql_lower:
                sql_query = self._insert_not_null_filter(sql_query, non_null_column)
                sql_lower = sql_query.lower()
        
        return sql_query
    
    def _insert_not_null_filter(self, sql_query: str, column: str) -> str:
        """Ensure queries referencing a column exclude NULL values."""
        sql_lower = sql_query.lower()
        if f"{column.lower()} is not null" in sql_lower:
            return sql_query
        
        clause_pattern = re.compile(r'\b(GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b', re.IGNORECASE)
        where_match = re.search(r'\bWHERE\b', sql_query, flags=re.IGNORECASE)
        
        if where_match:
            conditions_start = where_match.end()
            conditions = sql_query[conditions_start:]
            clause_match = clause_pattern.search(conditions)
            if clause_match:
                split_pos = clause_match.start()
                current_conditions = conditions[:split_pos].strip()
                remainder = conditions[split_pos:]
            else:
                current_conditions = conditions.strip()
                remainder = ""
            
            if current_conditions:
                new_conditions = f" {current_conditions} AND {column} IS NOT NULL "
            else:
                new_conditions = f" {column} IS NOT NULL "
            
            return f"{sql_query[:conditions_start]}{new_conditions}{remainder}"
        
        clause_match = clause_pattern.search(sql_query)
        if clause_match:
            insert_pos = clause_match.start()
        else:
            insert_pos = len(sql_query)
        
        before = sql_query[:insert_pos].rstrip()
        after = sql_query[insert_pos:]
        separator = "" if before.endswith((" ", "\n", "\t")) or not before else " "
        
        return f"{before}{separator}WHERE {column} IS NOT NULL {after}"
    
    def _execute_query_safely(self, sql_query: str, db_path: str, user_query: str = "") -> Optional[pd.DataFrame]:
        try:
            if not self.sqlite_conn:
                self.sqlite_conn = sqlite3.connect(db_path)
            
            results_df = pd.read_sql_query(sql_query, self.sqlite_conn)
            
            return results_df
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.debug(f"Failed query: {sql_query}")
            logger.debug(f"User query: {user_query}")
            return None
    
    def _generate_natural_language_response(
        self, 
        user_query: str, 
        sql_query: str, 
        results: pd.DataFrame,
        context: QueryContext
    ) -> str:
        if results is None:
            logger.warning(f"Query returned None - possible execution error. SQL: {sql_query[:200]}")
            return f"<p>Query execution failed. Please check if your question references valid columns and values.</p><p style='font-size: 0.9em; color: #94a3b8;'>Generated SQL: <code>{_html.escape(sql_query[:200])}</code></p>"
        
        if results.empty:
            logger.info(f"Query returned empty results. SQL: {sql_query[:200]}")
            diagnostic_info = self._get_diagnostic_info(user_query, context, sql_query)
            return f"<p>No results found for your query.</p>{diagnostic_info}"
        
        # detect record list queries vs aggregated analysis
        # show table directly for these instead of LLM narrative
        query_lower = user_query.lower()
        
        # "show all data" always gets table
        is_show_all_data = bool(re.search(
            r'\b(show|display|list|get|find|see)\s+(me\s+)?(all\s+)?data\b',
            query_lower
        )) or query_lower.strip() in ['show all data', 'show me all data', 'display all data', 'list all data', 'get all data']
        
        is_record_list_query = bool(re.search(
            r'\b(top\s+\d+|find\s+all|show\s+me|list|get|display|give)\s+.*?\b(vehicles?|records?|rows?|entries?|items?|vins?|vin\s+numbers?)\b',
            query_lower
        )) or bool(re.search(
            r'\b(give|show|list|get|find|display)\s+\d+\s+vins?\b',
            query_lower
        ))
        
        # "by X" queries show breakdown table
        is_breakdown_query = bool(re.search(
            r'\b(by|per|grouped by|group by)\s+\w+',
            query_lower
        ))
        
        # show all data = table only, no analysis
        if is_show_all_data and len(results) > 0:
            logger.info(f"Show all data query detected - showing table with {len(results)} results")
            return self._format_results_simple(results, user_query)
        
        # record list queries get table
        if is_record_list_query and len(results) > 0:
            logger.info(f"Record list query detected - showing table with {len(results)} results")
            return self._format_results_simple(results, user_query)
        
        # breakdown queries with multiple rows get table
        if is_breakdown_query and len(results) > 1:
            logger.info(f"Breakdown query detected - showing table with {len(results)} groups")
            return self._format_results_simple(results, user_query)
        
        try:
            logger.info(f"Generating natural language response for {len(results)} result(s)")
            
            natural_language_response = self._call_bedrock_for_natural_language(
                user_query, 
                sql_query, 
                results,
                context
            )
            
            if natural_language_response:
                logger.info("Natural language response generated successfully")
                return natural_language_response
            else:
                logger.warning("Natural language generation returned empty, using simple formatting")
                return self._format_results_simple(results, user_query)
                
        except Exception as e:
            logger.warning(f"Natural language generation failed: {e}, using simple formatting")
            return self._format_results_simple(results, user_query)
    
    def _call_bedrock_for_natural_language(
        self,
        user_query: str,
        sql_query: str,
        results: pd.DataFrame,
        context: QueryContext
    ) -> Optional[str]:
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            results_summary = self._prepare_results_summary(results, user_query)
            prompt = self._build_natural_language_prompt(user_query, sql_query, results_summary, results)
            
            bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
            model_id = config.model.bedrock_model_id
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            }
            
            response = bedrock.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            
            response_body = json.loads(response['body'].read())
            natural_response = response_body['content'][0]['text'].strip()
            
            validated_response = self._validate_and_clean_response(natural_response, results, user_query)
            
            return self._format_llm_response_as_html(validated_response, results)
            
        except ClientError as e:
            logger.error(f"Bedrock API error for natural language: {e}")
            return None
        except Exception as e:
            logger.error(f"Natural language generation failed: {e}", exc_info=True)
            return None
    
    def _prepare_results_summary(self, results: pd.DataFrame, user_query: str = "") -> str:
        summary_parts = []
        
        safe_results = self._filter_pii_from_dataframe(results.copy())
        safe_columns = safe_results.columns.tolist()
        
        query_intent = self._detect_query_intent(user_query)
        if query_intent == 'simple_lookup' and len(safe_results) > 10:
            relevant_columns = self._get_relevant_columns_for_query(user_query, safe_results)
            if relevant_columns and len(relevant_columns) < len(safe_columns):
                safe_results = safe_results[relevant_columns]
                safe_columns = relevant_columns
                summary_parts.append(f"NOTE: Showing only columns relevant to the query. Full dataset has {len(results.columns)} columns.")
        
        summary_parts.append(f"Query returned {len(results)} record(s) with {len(safe_columns)} column(s).")
        summary_parts.append(f"\nColumns: {', '.join(safe_columns)}")
        
        filtered_cols = set(results.columns) - set(safe_columns)
        if filtered_cols:
            summary_parts.append(f"\nNOTE: PII/sensitive columns filtered: {', '.join(filtered_cols)}")
        
        summary_parts.append("\n\nEXACT VALUES TO USE (use these exact numbers rounded to 2 decimal places):")
        if len(safe_results) == 1:
            for col in safe_columns:
                val = safe_results.iloc[0][col]
                formatted_val = self._format_number_to_2_decimals(val)
                summary_parts.append(f"  {col} = {formatted_val}")
        elif len(safe_results) <= 10:
            for idx, row in safe_results.iterrows():
                summary_parts.append(f"\nRecord {idx + 1}:")
                for col in safe_columns:
                    val = row[col]
                    formatted_val = self._format_number_to_2_decimals(val)
                    summary_parts.append(f"  {col} = {formatted_val}")
        else:
            if query_intent == 'simple_lookup':
                summary_parts.append(f"  Total count: {len(safe_results)} records")
                summary_parts.append("\nSample records (first 3):")
                for idx, row in safe_results.head(3).iterrows():
                    summary_parts.append(f"\nRecord {idx + 1}:")
                    for col in safe_columns[:5]:
                        val = row[col]
                        formatted_val = self._format_number_to_2_decimals(val)
                        summary_parts.append(f"  {col} = {formatted_val}")
            else:
                numeric_cols = safe_results.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    sum_val = self._format_number_to_2_decimals(safe_results[col].sum())
                    avg_val = self._format_number_to_2_decimals(safe_results[col].mean())
                    count_val = safe_results[col].count()
                    summary_parts.append(f"  {col}: SUM={sum_val}, AVG={avg_val}, COUNT={count_val}")
                summary_parts.append("\nSample records (first 3):")
                for idx, row in safe_results.head(3).iterrows():
                    summary_parts.append(f"\nRecord {idx + 1}:")
                    for col in safe_columns[:5]:
                        val = row[col]
                        formatted_val = self._format_number_to_2_decimals(val)
                        summary_parts.append(f"  {col} = {formatted_val}")
        
        if len(safe_results) == 1:
            summary_parts.append("\nSingle Record:")
            for col in safe_columns:
                value = safe_results.iloc[0][col]
                formatted_value = self._format_number_to_2_decimals(value)
                summary_parts.append(f"  - {col}: {formatted_value}")
        elif len(safe_results) <= 20:
            summary_parts.append("\nAll Records:")
            for idx, row in safe_results.iterrows():
                summary_parts.append(f"\nRecord {idx + 1}:")
                for col in safe_columns:
                    value = row[col]
                    formatted_value = self._format_number_to_2_decimals(value)
                    summary_parts.append(f"  - {col}: {formatted_value}")
        else:
            if query_intent == 'simple_lookup':
                summary_parts.append(f"\nTotal records: {len(safe_results)}")
                summary_parts.append("\nFirst 3 records (sample):")
                for idx, row in safe_results.head(3).iterrows():
                    summary_parts.append(f"\nRecord {idx + 1}:")
                    key_cols = [col for col in safe_columns if any(key in col.lower() for key in ['model', 'part', 'vehicle', 'id', 'name', 'type'])]
                    display_cols = key_cols[:5] if key_cols else safe_columns[:5]
                    for col in display_cols:
                        value = row[col]
                        formatted_value = self._format_number_to_2_decimals(value)
                        summary_parts.append(f"  - {col}: {formatted_value}")
            else:
                summary_parts.append("\nSummary Statistics:")
                summary_parts.append(f"Total records: {len(safe_results)}")
                summary_parts.append("\nFirst 5 records:")
                for idx, row in safe_results.head(5).iterrows():
                    summary_parts.append(f"\nRecord {idx + 1}:")
                    for col in safe_columns:
                        value = row[col]
                        formatted_value = self._format_number_to_2_decimals(value)
                        summary_parts.append(f"  - {col}: {formatted_value}")
            
            if query_intent != 'simple_lookup':
                numeric_cols = safe_results.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    summary_parts.append("\nNumeric Column Statistics:")
                    for col in numeric_cols:
                        min_val = self._format_number_to_2_decimals(safe_results[col].min())
                        max_val = self._format_number_to_2_decimals(safe_results[col].max())
                        mean_val = self._format_number_to_2_decimals(safe_results[col].mean())
                        summary_parts.append(f"  - {col}: min={min_val}, max={max_val}, mean={mean_val}")
        
        return "\n".join(summary_parts)
    
    def _format_number_to_2_decimals(self, value) -> str:
        import pandas as pd
        
        if pd.isna(value) or value is None:
            return "N/A"
        
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            return str(value)
        
        if num_value == int(num_value):
            return str(int(num_value))
        else:
            return f"{num_value:.2f}"
    
    def _filter_pii_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_safe = df.copy()
        
        columns_lower = {col.lower(): col for col in df_safe.columns}
        
        for pii_pattern in self.PII_COLUMNS:
            for col_lower, col_original in columns_lower.items():
                if pii_pattern.lower() in col_lower:
                    if col_original in df_safe.columns:
                        df_safe = df_safe.drop(columns=[col_original])
                        logger.info(f"Filtered PII column: {col_original}")
        
        return df_safe
    
    def _detect_query_intent(self, user_query: str) -> str:
        query_lower = user_query.lower()
        
        simple_keywords = ['find all', 'find', 'list', 'show all', 'show', 'get all', 'get', 
                          'which', 'what are', 'identify', 'locate']
        
        analysis_keywords = ['analyze', 'analysis', 'compare', 'trend', 'pattern', 'insight',
                            'why', 'how', 'explain', 'evaluate', 'assess', 'examine']
        
        aggregation_keywords = ['average', 'avg', 'mean', 'total', 'sum', 'count', 'rate', 
                               'percentage', 'percent', 'statistics', 'stats']
        
        if any(keyword in query_lower for keyword in simple_keywords):
            if any(keyword in query_lower for keyword in aggregation_keywords):
                return 'aggregation'
            return 'simple_lookup'
        
        if any(keyword in query_lower for keyword in analysis_keywords):
            return 'analysis'
        
        if any(keyword in query_lower for keyword in aggregation_keywords):
            return 'aggregation'
        
        return 'simple_lookup'
    
    def _get_relevant_columns_for_query(self, user_query: str, results: pd.DataFrame) -> List[str]:
        query_lower = user_query.lower()
        all_columns = results.columns.tolist()
        
        key_columns = ['model', 'part', 'vehicle', 'vin', 'id', 'name', 'type', 'category']
        
        query_keywords = []
        if 'battery' in query_lower:
            query_keywords.extend(['battery', 'voltage', 'charge', 'degradation'])
        if 'engine' in query_lower:
            query_keywords.extend(['engine', 'rpm', 'coolant', 'oil', 'load'])
        if 'brake' in query_lower:
            query_keywords.extend(['brake', 'pad', 'pressure', 'torque'])
        if 'mileage' in query_lower or 'miles' in query_lower:
            query_keywords.extend(['mileage', 'miles', 'odometer'])
        if 'failure' in query_lower or 'fail' in query_lower:
            query_keywords.extend(['failure', 'fail', 'claim', 'repair'])
        
        relevant_columns = []
        for col in all_columns:
            col_lower = col.lower()
            if any(key in col_lower for key in key_columns):
                relevant_columns.append(col)
            elif any(keyword in col_lower for keyword in query_keywords):
                relevant_columns.append(col)
        
        return relevant_columns if relevant_columns else all_columns
    
    def _build_natural_language_prompt(
        self,
        user_query: str,
        sql_query: str,
        results_summary: str,
        results: pd.DataFrame
    ) -> str:
        query_intent = self._detect_query_intent(user_query)
        
        if query_intent == 'simple_lookup':
            focus_instruction = """
**CRITICAL - SIMPLE LOOKUP QUERY**: The user asked a simple "find/list" question. Your response should:
1. **Answer Only**: Provide a clear, concise answer to the specific question asked
2. **Count/Summary**: State the count or number of records found
3. **Key Information Only**: Only mention information that directly relates to what was asked (e.g., if asked about "battery failures", only mention battery-related data, not engine/brake/geographic data)
4. **No Unrelated Statistics**: DO NOT include averages, statistics, or details about columns that are not directly relevant to the question
5. **Keep It Brief**: This is a lookup query, not an analysis - be concise and focused

EXAMPLE for "Find all Sentra vehicles with battery failures":
- GOOD: "Found 246 Sentra vehicles with battery failures where mileage is between 0 and 50,000 miles."
- BAD: "Found 246 vehicles. The average model year is 2023.03. The total claims are 35. The average battery voltage is 12.25V..." (too much unrelated info)
"""
        elif query_intent == 'aggregation':
            focus_instruction = """
**AGGREGATION QUERY**: The user asked for statistics, averages, totals, or rates. Your response should:
1. **Answer**: Start with the specific statistic requested (average, total, rate, etc.)
2. **Relevant Statistics Only**: Include only statistics that directly relate to the question
3. **Context**: Provide brief context about what the numbers mean
4. **No Unrelated Aggregations**: DO NOT calculate or mention statistics for columns not mentioned in the question
"""
        else:
            focus_instruction = """
**ANALYSIS QUERY**: The user asked for analysis, comparison, or insights. Your response should:
1. **Answer**: Start with key findings
2. **Context & Insights**: Provide meaningful context and analytical observations
3. **Patterns**: Highlight patterns visible in the data
4. **Relevant Analysis Only**: Focus analysis on what was asked, not all available columns
"""
        
        # Check if results indicate no data (empty, N/A, or 0 records)
        has_no_data = (
            results.empty or 
            len(results) == 0 or
            'returned 0 record' in results_summary.lower() or
            'no records' in results_summary.lower() or
            (len(results) == 1 and any(
                str(results.iloc[0][col]).upper() in ['N/A', 'NULL', 'NONE', '0', '0.00', '0.0'] 
                for col in results.columns 
                if pd.api.types.is_numeric_dtype(results[col]) or results[col].dtype == 'object'
            ))
        )
        
        no_data_instruction = ""
        if has_no_data:
            no_data_instruction = """
**CRITICAL - NO DATA FOUND**:
- The query returned NO RESULTS or indicates NO DATA (N/A, 0, NULL, etc.)
- Keep the response to a single factual sentence
- **DO NOT include Industry Context, Issues/Concerns, or Recommendations**
- Simply state that no data was found for the requested query
- Example: "<p>No data found for [the requested item]; the query returned no records.</p>"
- Keep it brief and factual - no analysis needed when there's no data to analyze
"""
        
        prompt = f"""You are an expert automotive telematics analyst specializing in vehicle failure analysis, predictive maintenance, and fleet reliability. A user asked a question about vehicle telematics data, and a SQL query was executed to retrieve the answer.

USER'S QUESTION: {user_query}

SQL QUERY EXECUTED: {sql_query}

QUERY RESULTS:
{results_summary}

{no_data_instruction}

{focus_instruction}

CRITICAL REQUIREMENTS - READ CAREFULLY:

1. **NO HALLUCINATION - DATA ACCURACY**: 
   - You MUST use ONLY the exact numbers, values, and data provided in QUERY RESULTS above
   - DO NOT make up, estimate, or infer any numbers that are not explicitly in the results
   - DO NOT reference data that is not in the QUERY RESULTS
   - If a number is not in the results, DO NOT mention it
   - DO NOT add percentages, ratios, or calculations that are not in the results
   - **EXCEPTION**: You MAY reference general automotive industry standards/benchmarks for context, but MUST clearly label them as "industry standard" or "typical benchmark" and distinguish them from the actual data

2. **ACCURACY REQUIREMENTS**:
   - All numbers in your response MUST match the values in "EXACT VALUES TO USE" section
   - All decimal numbers MUST be rounded to exactly 2 decimal places (e.g., 0.22, 21.88, 245.00)
   - Percentages should be displayed with 2 decimal places (e.g., 21.88%, not 21.875%)
   - Whole numbers can be displayed without decimals (e.g., 245, not 245.00)
   - Verify every number you mention exists in the QUERY RESULTS (except industry benchmarks which should be clearly labeled)

3. **PII PROTECTION**:
   - VIN numbers, email addresses, and phone numbers have been filtered from the data
   - DO NOT mention any VIN numbers, email addresses, phone numbers, or personal identifiers
   - DO NOT reference customer names, contact information, or any PII
   - Focus only on aggregated, anonymized data
   - NOTE: All numeric values (percentages, counts, rates) should be displayed - these are NOT PII

4. **ANALYST-FRIENDLY ENHANCEMENTS** (while maintaining accuracy):
   - **Industry Context**: When appropriate, provide industry standard benchmarks for comparison (e.g., "The failure rate of 21.88% is above the typical industry benchmark of 10-15% for similar vehicles")
   - **Issue Detection**: If figures seem unusual, inconsistent, or problematic, point them out professionally (e.g., "The battery voltage of 8.5V is critically low and indicates potential battery failure")
   - **Recommendations**: Provide actionable recommendations based on the data (e.g., "Given the high failure rate, recommend enhanced monitoring and proactive maintenance")
   - **Professional Analysis**: Use automotive analyst terminology and provide meaningful context

TASK: Generate a professional, automotive analyst-friendly response that DIRECTLY answers the user's question. Your response should:

1. **Opening Statement**: Start with a clear, direct sentence using EXACT numbers from QUERY RESULTS. 
   - **DO NOT** prefix the sentence with labels such as "Answer:" or meta phrases like "According to the SQL query results"
   - **DO NOT** reference SQL queries, databases, or technical implementation details
   - Simply state the answer directly as if you're an analyst reporting findings
   - **CRITICAL**: If the user asks for specific records (e.g., "Give 3 VINs", "List 5 vehicles"), you MUST include the actual VINs/identifiers in your answer. Do NOT just say "The query returned 3 VINs" - you must list them.
   - Example: "The failure rate for Sentra is **21.88%**" (NOT "According to the SQL query results, the failure rate for Sentra is 21.88%")
   - Example for VIN queries: "Here are 3 VINs with DTC codes: **1N4AZ1CP0JC123456**, **1N4AZ1CP0JC123457**, **1N4AZ1CP0JC123458**" (NOT "The SQL query returned 3 VIN numbers with associated DTC codes")
2. **Data Analysis**: 
   - Present the data clearly and professionally
   - If asked to "find vehicles", provide the count and key identifying info
   - If asked about specific metrics (failures, rates, etc.), focus on those metrics
   - DO NOT include statistics about unrelated columns unless specifically asked
3. **Industry Context** (when appropriate):
   - Compare results to typical industry benchmarks when relevant (e.g., failure rates, battery health, etc.)
   - Clearly label industry benchmarks as "industry standard" or "typical benchmark"
   - Use phrases like "compared to the typical industry benchmark of X%" or "industry standard range is typically X-Y"
4. **Issue Detection** (when applicable):
   - Point out unusual, concerning, or inconsistent figures
   - Use professional language (e.g., "notably high", "concerning", "requires attention", "critically low")
   - Explain why the figure is problematic based on automotive knowledge
   - If data is normal/healthy/within acceptable ranges, do NOT create issues where none exist
5. **Recommendations** (ONLY when Issues/Concerns are identified):
   - **CRITICAL**: Only include Recommendations if you have identified actual Issues/Concerns in step 4
   - If data is normal, healthy, or within acceptable industry benchmarks, DO NOT include Recommendations
   - Recommendations should directly address the specific issues identified
   - Focus on practical next steps (e.g., "recommend investigation", "suggest enhanced monitoring", "consider proactive maintenance")
   - Keep recommendations specific and relevant to the identified problems
6. **Professional Terminology**: Use appropriate automotive/telematics terminology throughout
7. **Clarity**: Keep the response clear, well-structured, and easy to understand for an automotive analyst

FORMATTING REQUIREMENTS:
- Use HTML formatting: <p>, <strong>, <ul>, <li>, <em>
- The first paragraph MUST be a single `<p>...</p>` sentence summarizing the answer (no labels like "Answer:")
- If you include Industry Context, Issues/Concerns, or Recommendations, each MUST be in its own paragraph starting with `<p><strong>Industry Context:</strong> ...</p>`, `<p><strong>Issues/Concerns:</strong> ...</p>`, `<p><strong>Recommendations:</strong></p>` followed by a list (if multiple items).
- Structure with clear sections: Answer, Industry Context (if applicable), Issues/Concerns (if applicable), Recommendations (ONLY if Issues/Concerns are identified)
- **Bold formatting should be used SPARINGLY** - only for:
  * Key numbers/statistics (e.g., "Found <strong>246</strong> vehicles", "failure rate of <strong>21.88%</strong>")
  * Important keywords (e.g., model names, failure types, critical values)
  * NOT for entire sentences or paragraphs
  * NOT for common words like "the", "is", "are", etc.
- Use bullet points for lists of findings, recommendations, or multiple issues
- Include percentages, ratios, and comparisons clearly
- When providing industry context, use phrases like "compared to the typical industry benchmark" or "industry standard range"
- When pointing out issues, use professional language: "notably high", "concerning", "requires attention", "critically low"
- When providing recommendations, structure them clearly (numbered or bulleted)
- IMPORTANT: Use single spacing between paragraphs. Do NOT add multiple blank lines or excessive spacing. Keep formatting clean and compact.
- IMPORTANT: Do NOT wrap entire sentences or paragraphs in <strong> tags. Only bold specific numbers and important keywords.

EXAMPLE STYLES (using EXACT numbers from results, rounded to 2 decimals):

**Simple Lookup Query**: "Find all Sentra vehicles with battery failures where mileage is between 0 and 50000"
- GOOD: "Found <strong>246</strong> Sentra vehicles with battery failures where mileage is between 0 and 50,000 miles."
- BAD: "Found 246 vehicles. The average model year is 2023.03. Total claims are 35. Average battery voltage is 12.25V..." (includes unrelated statistics)

**Aggregation Query with Industry Context**: "What's the failure rate for Sentra?"
- GOOD: "The failure rate for the Sentra model is <strong>21.88%</strong>, based on 245 failures across 1,120 records. This is notably higher than the typical industry benchmark of 10-15% for similar sedan models, indicating potential reliability concerns that warrant further investigation."
- GOOD: "The failure rate is <strong>0.22</strong> (<strong>21.88%</strong>), based on 245 failures across 1,120 records. Compared to the industry standard range of 10-15% for vehicles in this class, this elevated rate suggests the need for enhanced monitoring and proactive maintenance strategies."
- BAD: "According to the SQL query results, the failure rate for Sentra is 21.88%..." (Don't mention SQL or data sources)

**Analysis Query with Issue Detection**: "Analyze battery performance for Sentra vehicles"
- GOOD: "Analysis of Sentra vehicles shows an average battery state of charge of <strong>79.31%</strong> and average battery voltage of <strong>12.25V</strong>. The average battery degradation is <strong>7.47%</strong>. The voltage reading of 12.25V is within normal operating range (typically 12.4-12.7V for healthy batteries), though the degradation rate of 7.47% may require attention depending on vehicle age and mileage."
- GOOD: "Analysis reveals an average battery voltage of <strong>8.5V</strong>, which is critically low (normal range: 12.4-12.7V). This indicates potential battery failure and requires immediate attention. Recommend diagnostic testing and battery replacement for affected vehicles."

**Query with Issues and Recommendations**: "What's the failure rate for Sentra?"
- GOOD: "The failure rate for the Sentra model is <strong>21.88%</strong>, which exceeds the typical industry benchmark of 10-15%. <strong>Issues/Concerns:</strong> This elevated rate indicates potential reliability concerns. <strong>Recommendations:</strong> (1) Conduct root cause analysis to identify common failure modes, (2) Implement enhanced monitoring for Sentra vehicles, (3) Consider proactive maintenance schedules to reduce failure incidents."

**Query with Normal/Healthy Data (NO Recommendations)**: "What's the failure rate for Leaf?"
- GOOD: "The failure rate for the Leaf model is <strong>8.5%</strong>, based on 95 failures across 1,118 records. This is within the typical industry benchmark of 10-15% for similar electric vehicles, indicating normal reliability performance."
- BAD: "The failure rate is 8.5%... <strong>Recommendations:</strong> Continue monitoring..." (No issues identified, so no recommendations needed)

KEY PRINCIPLES:
- All decimal values must be rounded to 2 decimal places
- Use EXACT numbers from results - never estimate or approximate
- **Never mention SQL queries, databases, or data sources** - answer directly as if reporting findings
- **Never start with meta-commentary** like "According to the SQL query results", "Based on the data", "The query returned" - state the answer directly
- Industry benchmarks should be clearly labeled and distinguished from actual data
- Issue detection should be professional and based on automotive knowledge
- **Recommendations should ONLY appear when Issues/Concerns are identified** - if data is normal/healthy, skip recommendations entirely
- Recommendations should be specific, actionable, and directly address the identified problems
- Only include information directly relevant to the question

VALIDATION CHECKLIST (before responding):
- Every number from the data exists in the "EXACT VALUES TO USE" section above
- I have not made up any numbers - all data numbers come from results
- I have clearly labeled any industry benchmarks as "industry standard" or "typical benchmark"
- I have not mentioned any VIN, email, phone, or PII
- I have NOT started with meta-commentary like "According to the SQL query results", "Based on the data", "The query returned", etc. - I stated the answer directly
- I have NOT mentioned SQL queries, databases, or technical implementation details
- I am only discussing information that DIRECTLY relates to the user's question
- I have NOT included statistics or details about columns not mentioned in the question (unless for context)
- All decimal numbers are rounded to exactly 2 decimal places (e.g., 0.22, 21.88, 245.00)
- Percentages are displayed with 2 decimal places (e.g., 21.88%, not 21.875%)
- Whole numbers can be displayed without decimals (e.g., 245)
- I am using the numbers from "EXACT VALUES TO USE" section (already formatted to 2 decimals)
- My response is professional, analyst-friendly, and provides value (context, issues, recommendations when appropriate)
- I have pointed out any concerning or unusual figures in the data (or noted if data is normal/healthy)
- I have ONLY included Recommendations if I identified actual Issues/Concerns - if data is normal/healthy, I did NOT include Recommendations
- My response is clear, well-structured, and easy to understand

CRITICAL: 
- The "EXACT VALUES TO USE" section above contains the ONLY data numbers you are allowed to use, already formatted to 2 decimal places
- Industry benchmarks can be referenced for context but MUST be clearly labeled as such
- All analysis, issue detection, and recommendations must be based on the actual data provided
- Never hallucinate or make up data - be professional and accurate

RESPONSE (HTML formatted, professional automotive analyst style, using ONLY exact values from "EXACT VALUES TO USE" section):"""
        
        return prompt
    
    def _validate_and_clean_response(
        self, 
        response: str, 
        results: pd.DataFrame,
        user_query: str
    ) -> str:
        response = self._remove_obvious_pii_from_text(response)
        
        # Check if results indicate no data (empty or contains N/A/null values)
        has_no_data = (
            results.empty or 
            len(results) == 0 or
            (len(results) == 1 and any(
                str(results.iloc[0][col]).upper() in ['N/A', 'NULL', 'NONE', '0', '0.00', '0.0', ''] 
                for col in results.columns 
                if pd.api.types.is_numeric_dtype(results[col]) or results[col].dtype == 'object'
            ))
        )
        
        # If no data, remove Industry Context, Issues/Concerns, and Recommendations sections
        if has_no_data:
            # Remove Industry Context section (handles multiline content)
            response = re.sub(
                r'<p><strong>Industry\s+Context:</strong>.*?(?=<p><strong>|</p>\s*<p>|$)',
                '',
                response,
                flags=re.DOTALL | re.IGNORECASE
            )
            # Remove Issues/Concerns section
            response = re.sub(
                r'<p><strong>Issues/Concerns:</strong>.*?(?=<p><strong>|</p>\s*<p>|$)',
                '',
                response,
                flags=re.DOTALL | re.IGNORECASE
            )
            # Remove Recommendations section
            response = re.sub(
                r'<p><strong>Recommendations:</strong>.*?(?=<p><strong>|</p>\s*<p>|$)',
                '',
                response,
                flags=re.DOTALL | re.IGNORECASE
            )
            # Clean up any extra whitespace or empty paragraphs
            response = re.sub(r'<p>\s*</p>', '', response)
            response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        
        safe_results = self._filter_pii_from_dataframe(results.copy())
        
        numeric_pattern = r'\b\d+\.?\d*%?\b'
        numbers_in_response = re.findall(numeric_pattern, response)
        
        actual_values = self._extract_actual_values_from_results(safe_results)
        key_values = self._extract_key_aggregated_values(safe_results)
        actual_values.extend(key_values)
        
        hallucination_detected = False
        for num_str in numbers_in_response:
            num_clean = num_str.replace('%', '').strip()
            try:
                num_value = float(num_clean)
                if not self._number_exists_in_results(num_value, actual_values, tolerance=0.1):
                    logger.warning(f"Potential hallucination detected: Number '{num_str}' in response may not match results. Query: {user_query[:50]}")
                    hallucination_detected = True
            except ValueError:
                pass
        
        response = self._format_decimals_to_2_places(response)
        
        if hallucination_detected and config.debug:
            response = f"<p style='color: orange;'><em>Note: Some numbers in this response may need verification against source data.</em></p>{response}"
        
        return response
    
    def _format_decimals_to_2_places(self, text: str) -> str:
        def format_decimal(match):
            num_str = match.group(0)
            has_percent = '%' in num_str
            num_clean = num_str.replace('%', '').strip()
            
            try:
                num_value = float(num_clean)
                if abs(num_value - int(num_value)) < 0.001:
                    formatted = str(int(round(num_value)))
                else:
                    formatted = f"{num_value:.2f}"
                
                if has_percent:
                    formatted += "%"
                
                return formatted
            except ValueError:
                return num_str
        
        decimal_pattern = r'(?<![\d\.])(\d+\.\d{3,})(%?)(?![\d])'
        text = re.sub(decimal_pattern, format_decimal, text)
        
        percent_pattern = r'(?<![\d\.])(\d+\.\d{3,})%(?![\d])'
        text = re.sub(percent_pattern, format_decimal, text)
        
        return text
    
    def _remove_obvious_pii_from_text(self, text: str) -> str:
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL REDACTED]', text)
        text = re.sub(r'\b(?<!\.\d)(?<!\d\.)\d{3}[-.]?\d{3}[-.]?\d{4}(?!\.\d)\b', '[PHONE REDACTED]', text)
        text = re.sub(r'\b(?<!\.\d)(?<!\d\.)\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\.\d)\b', '[PHONE REDACTED]', text)
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REDACTED]', text)
        
        return text
    
    def _extract_key_aggregated_values(self, results: pd.DataFrame) -> List[float]:
        values = []
        
        numeric_cols = results.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            try:
                values.append(float(results[col].sum()))
            except (ValueError, TypeError):
                pass
            try:
                values.append(float(results[col].mean()))
            except (ValueError, TypeError):
                pass
            try:
                values.append(float(results[col].count()))
            except (ValueError, TypeError):
                pass
            try:
                values.append(float(results[col].min()))
                values.append(float(results[col].max()))
            except (ValueError, TypeError):
                pass
        
        if 'failures_count' in numeric_cols and len(results) > 0:
            total_failures = results['failures_count'].sum()
            total_records = len(results)
            if total_records > 0:
                failure_rate = (total_failures / total_records) * 100
                values.append(failure_rate)
        
        return values
    
    def _extract_actual_values_from_results(self, results: pd.DataFrame) -> List[float]:
        values = []
        
        numeric_cols = results.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            for val in results[col].dropna():
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    pass
        
        for col in results.select_dtypes(include=['object']).columns:
            for val in results[col].dropna():
                numbers = re.findall(r'\d+\.?\d*', str(val))
                for num_str in numbers:
                    try:
                        values.append(float(num_str))
                    except ValueError:
                        pass
        
        return values
    
    def _number_exists_in_results(self, number: float, actual_values: List[float], tolerance: float = 0.01) -> bool:
        for actual_val in actual_values:
            if abs(number - actual_val) < tolerance or abs(number - actual_val) / max(abs(actual_val), 1) < tolerance:
                return True
        return False
    
    def _format_llm_response_as_html(self, llm_response: str, results: pd.DataFrame) -> str:
        response = llm_response.strip()
        
        def clean_bold_tags(match):
            content = match.group(1).strip()
            
            if re.match(r'^[\d,.\s%]+$', content):
                return match.group(0)
            
            words = content.split()
            if len(words) <= 2 and not re.search(r'[.!?]', content):
                common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'this', 'that', 
                               'these', 'those', 'found', 'there', 'with', 'where', 'and', 'or',
                               'for', 'from', 'to', 'in', 'on', 'at', 'by', 'of'}
                if len(words) == 1 and words[0].lower() not in common_words:
                    return match.group(0)
                elif len(words) == 2 and words[0].lower() not in common_words:
                    return match.group(0)
            
            return content
        
        response = re.sub(r'<strong>([^<]+)</strong>', clean_bold_tags, response)
        
        common_starters = ['Found', 'The', 'There', 'This', 'These', 'That', 'Based on', 
                          'According to', 'Analysis shows', 'Results show', 'Data shows']
        for starter in common_starters:
            pattern = rf'<strong>\s*{re.escape(starter)}\s*</strong>'
            response = re.sub(pattern, starter, response, flags=re.IGNORECASE)
        
        response = re.sub(r'<strong>\s*<strong>([^<]+)</strong>\s*</strong>', r'<strong>\1</strong>', response)
        
        response = re.sub(r'\n{2,}', '\n', response)
        response = re.sub(r' {2,}', ' ', response)
        
        if '<p>' in response or '<div>' in response or '<ul>' in response:
            response = re.sub(r'</p>\s*\n\s*<p>', '</p><p>', response)
            response = re.sub(r'</div>\s*\n\s*<div>', '</div><div>', response)
            response = re.sub(r'</li>\s*\n\s*<li>', '</li><li>', response)
            response = re.sub(r'<p>\s*\n\s*', '<p>', response)
            response = re.sub(r'\s*\n\s*</p>', '</p>', response)
            response = re.sub(r'<p>\s*</p>', '', response)
            response = re.sub(r'</p>\s*<p>', '</p> <p>', response)
        else:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            paragraphs = []
            current_para = []
            for line in lines:
                if line and not line.startswith(('•', '-', '*')):
                    current_para.append(line)
                else:
                    if current_para:
                        paragraphs.append(' '.join(current_para))
                        current_para = []
                    paragraphs.append(line)
            if current_para:
                paragraphs.append(' '.join(current_para))
            
            response = ' '.join([f'<p>{p}</p>' for p in paragraphs if p])
        
        response = re.sub(r'\s+', ' ', response)
        response = re.sub(r'</p>\s+<p>', '</p><p>', response)
        
        css_style = '''<style>
        .text-to-sql-response p {
            margin: 0 0 12px 0;
            padding: 0;
            line-height: 1.6;
        }
        .text-to-sql-response p:last-child {
            margin-bottom: 0;
        }
        .text-to-sql-response ul {
            margin: 8px 0 12px 20px;
            padding: 0;
        }
        .text-to-sql-response li {
            margin: 4px 0;
            padding: 0;
        }
        </style>'''
        
        response = f'{css_style}<div class="text-to-sql-response">{response}</div>'
        
        return response
    
    def _extract_metric_from_query(self, user_query: str, results: pd.DataFrame) -> str:
        """Extract what metric is being asked about from the query."""
        query_lower = user_query.lower()
        
        # Check result columns for clues about what's being calculated
        result_cols_lower = [str(col).lower() for col in results.columns]
        
        # Look for metric keywords in query
        if re.search(r'\baverage\s+(engine\s+)?rpm\b', query_lower):
            return "Average engine RPM"
        elif re.search(r'\baverage\s+mileage\b', query_lower):
            return "Average mileage"
        elif re.search(r'\baverage\s+battery\s+voltage\b', query_lower):
            return "Average battery voltage"
        elif re.search(r'\baverage\s+battery\s+soc\b', query_lower):
            return "Average battery SOC"
        elif re.search(r'\baverage\s+(\w+)', query_lower):
            match = re.search(r'\baverage\s+(\w+(?:\s+\w+)?)', query_lower)
            if match:
                metric = match.group(1).title()
                return f"Average {metric}"
        
        # Check for count/total/sum queries
        if re.search(r'\b(failure\s+)?count\b', query_lower) or any('count' in col for col in result_cols_lower):
            return "Failure count"
        elif re.search(r'\btotal\s+failures\b', query_lower) or any('total' in col and 'failure' in col for col in result_cols_lower):
            return "Total failures"
        elif re.search(r'\bsum\s+of\s+(\w+)', query_lower):
            match = re.search(r'\bsum\s+of\s+(\w+)', query_lower)
            if match:
                metric = match.group(1).title()
                return f"Sum of {metric}"
        
        # Check result columns for common patterns
        for col in results.columns:
            col_lower = str(col).lower()
            if 'avg' in col_lower or 'average' in col_lower:
                # Extract the metric name from column (e.g., "avg_engine_rpm" -> "Average engine RPM")
                metric_name = col_lower.replace('avg_', '').replace('average_', '').replace('_', ' ').title()
                return f"Average {metric_name}"
            elif 'sum' in col_lower or 'total' in col_lower:
                metric_name = col_lower.replace('sum_', '').replace('total_', '').replace('_', ' ').title()
                return f"Total {metric_name}"
            elif 'count' in col_lower:
                metric_name = col_lower.replace('count', '').replace('_', ' ').strip().title()
                if metric_name:
                    return f"{metric_name} count"
                return "Count"
        
        # Default fallback
        return "Results"
    
    def _format_results_simple(self, results: pd.DataFrame, user_query: str = "") -> str:
        if len(results) == 1 and len(results.columns) == 1:
            value = _html.escape(str(results.iloc[0, 0]))
            return f"<p>{value}</p>"
        
        # For vehicle list queries or VIN-specific queries, show only VIN column if available
        query_lower = (user_query or "").lower()
        is_vehicle_list = bool(re.search(r'\b(vehicles?|vehicle\s+records?)\b', query_lower))
        is_vin_query = bool(re.search(r'\b(vins?|vin\s+numbers?|vehicle\s+identification\s+numbers?)\b', query_lower))
        wants_model = bool(re.search(r'\bmodel(s)?\b', query_lower))
        
        display_results = results.copy()
        if is_vehicle_list or is_vin_query:
            # Find VIN column (check common variations)
            vin_col = None
            model_col = None
            for col in display_results.columns:
                col_lower = str(col).lower()
                if not vin_col and col_lower in ['vin', 'vehicle_identification_number', 'vehicle_id']:
                    vin_col = col
                if not model_col and col_lower in ['model', 'vehicle_model', 'model_name']:
                    model_col = col
                if vin_col and model_col:
                    break
            
            if vin_col:
                cols_to_show = [vin_col]
                if wants_model and model_col and model_col not in cols_to_show:
                    cols_to_show.append(model_col)
                display_results = display_results[cols_to_show]
        
        # build answer header
        count = len(results)
        query_lower = (user_query or "").lower()
        
        # handle "show all data" queries
        is_show_all_data = bool(re.search(
            r'\b(show|display|list|get|find|see)\s+(me\s+)?(all\s+)?data\b',
            query_lower
        )) or query_lower.strip() in ['show all data', 'show me all data', 'display all data', 'list all data', 'get all data']
        
        if is_show_all_data:
            if count == 1:
                answer_text = f"Showing <strong>1 record</strong> from the dataset."
            elif count <= 10:
                answer_text = f"Showing <strong>{count} records</strong> from the dataset."
            else:
                answer_text = f"Showing <strong>{len(results)} records</strong> from the dataset (displaying first 10):"
        else:
            # extract metric from query
            metric_text = self._extract_metric_from_query(user_query, results)
            
            # breakdown queries like "by city", "by model"
            breakdown_match = re.search(r'\b(by|per)\s+(\w+)', query_lower)
            if breakdown_match:
                breakdown_dimension = breakdown_match.group(2)
                if count == 1:
                    answer_text = f"{metric_text} <strong>by {breakdown_dimension}</strong>:"
                else:
                    answer_text = f"{metric_text} <strong>by {breakdown_dimension}</strong> ({count} {breakdown_dimension}s):"
            else:
                # regular query
                if count == 1:
                    answer_text = f"Found <strong>1 record</strong> matching your query."
                elif count <= 10:
                    answer_text = f"Found <strong>{count} records</strong> matching your query."
                else:
                    answer_text = f"Found <strong>{len(results)} records</strong>. Showing first 10:"
        
        html_table = display_results.head(10).to_html(classes='table table-striped', table_id='query-results', escape=False, index=False)
        # clean up pandas HTML whitespace
        html_table = re.sub(r'\s+', ' ', html_table)
        return f"<p style='margin-bottom: 8px;'>{answer_text}</p><div style='margin-top: 0;'>{html_table}</div>"
    
    def _get_diagnostic_info(self, user_query: str, context: QueryContext, sql_query: str = "") -> str:
        """Provide diagnostic information when query returns no results."""
        query_lower = user_query.lower()
        df = context.df_history
        
        diagnostic_parts = []
        diagnostic_parts.append("<p style='font-size: 0.9em; color: #94a3b8; margin-top: 15px;'><strong>Diagnostic Information:</strong></p>")
        diagnostic_parts.append("<ul style='font-size: 0.9em; color: #94a3b8;'>")
        
        model_match = re.search(r'\b(leaf|sentra|ariya|altima|rogue|pathfinder)\b', query_lower)
        if model_match:
            model_name = model_match.group(1).capitalize()
            if 'model' in df.columns:
                model_count = len(df[df.get('model', pd.Series(dtype=str)).str.lower() == model_name.lower()])
                diagnostic_parts.append(f"<li>Total {model_name} vehicles in dataset: {model_count}</li>")
        
        if 'critical' in query_lower or 'health' in query_lower:
            health_cols = [col for col in df.columns if 'health' in col.lower() or 'status' in col.lower()]
            if health_cols:
                for col in health_cols[:2]:
                    unique_values = df[col].dropna().unique()[:5]
                    diagnostic_parts.append(f"<li>Available {col} values: {', '.join([str(v) for v in unique_values])}</li>")
        
        if 'battery' in query_lower and 'voltage' in query_lower:
            voltage_cols = [col for col in df.columns if 'voltage' in col.lower() and 'battery' in col.lower()]
            if voltage_cols:
                for col in voltage_cols[:1]:
                    if df[col].dtype in ['float64', 'int64']:
                        min_val = df[col].min()
                        max_val = df[col].max()
                        avg_val = df[col].mean()
                        diagnostic_parts.append(f"<li>Battery voltage range: {min_val:.2f} to {max_val:.2f} (avg: {avg_val:.2f})</li>")
        
        if 'temperature' in query_lower and 'battery' in query_lower:
            temp_cols = [col for col in df.columns if 'temperature' in col.lower() and 'battery' in col.lower()]
            if temp_cols:
                for col in temp_cols[:1]:
                    if df[col].dtype in ['float64', 'int64']:
                        min_val = df[col].min()
                        max_val = df[col].max()
                        avg_val = df[col].mean()
                        diagnostic_parts.append(f"<li>Battery temperature range: {min_val:.2f} to {max_val:.2f} (avg: {avg_val:.2f})</li>")
        
        diagnostic_parts.append("</ul>")
        
        if sql_query:
            diagnostic_parts.append(f"<p style='font-size: 0.85em; color: #64748b; margin-top: 10px;'><strong>Generated SQL:</strong> <code style='font-size: 0.85em;'>{_html.escape(sql_query[:300])}</code></p>")
        
        diagnostic_parts.append("<p style='font-size: 0.85em; color: #64748b; margin-top: 10px;'>Tip: Try adjusting your filter criteria or check if the column names match your query.</p>")
        
        return "".join(diagnostic_parts)
    
    def _cleanup_database(self):
        if self.sqlite_conn:
            try:
                self.sqlite_conn.close()
            except (sqlite3.Error, AttributeError):
                pass
            self.sqlite_conn = None
