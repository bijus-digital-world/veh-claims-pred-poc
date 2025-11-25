# Complex Test Questions for Text-to-SQL System

## Category 1: Multi-Condition Filtering

1. **"Find all Sentra vehicles with battery failures where mileage is between 30000 and 50000, manufactured in 2022 or 2023"**
   - Tests: Multiple WHERE conditions, BETWEEN, OR logic, date filtering

2. **"Show me all Leaf vehicles with critical health status where battery voltage is less than 12 OR battery temperature is greater than 60"**
   - Tests: OR conditions, multiple sensor thresholds

3. **"Find vehicles where model is Sentra AND (primary_failed_part is Battery OR primary_failed_part is Clutch) AND mileage is greater than 50000"**
   - Tests: Complex AND/OR nesting, multiple filters

4. **"List all vehicles with battery failures where mileage is between 0 and 50000 AND battery_voltage is less than 12 AND battery_soc is below 20%"**
   - Tests: Multiple AND conditions, range queries, percentage thresholds

5. **"Find all Ariya vehicles where failures_count is greater than 2 AND predicted_claim_probability is above 0.3 AND supplier_quality_score is below 80"**
   - Tests: Multiple numeric conditions, quality metrics

## Category 2: Complex Aggregations

6. **"What's the average failure rate by model for vehicles with mileage between 0 and 50000?"**
   - Tests: GROUP BY with WHERE filter, calculated rates

7. **"Show me the total failures, average mileage, and maximum battery voltage grouped by model and primary_failed_part"**
   - Tests: Multiple aggregations, multiple GROUP BY columns

8. **"What's the failure rate by model for vehicles manufactured in 2022, showing only models with more than 10 failures?"**
   - Tests: GROUP BY, HAVING clause, date filtering

9. **"Calculate the average battery degradation by model for vehicles where battery_voltage is between 12 and 13"**
   - Tests: Conditional aggregation, range filtering

10. **"What's the total number of failures, average predicted claim probability, and maximum mileage grouped by model and year?"**
    - Tests: Multiple aggregations, date extraction for grouping

## Category 3: Statistical Analysis

11. **"What's the standard deviation of battery voltage for Sentra vehicles?"**
    - Tests: Statistical functions (if supported)

12. **"Show me the median mileage for each model where failures_count is greater than 0"**
    - Tests: Median calculation, conditional grouping

13. **"What's the 90th percentile of battery degradation for vehicles with battery failures?"**
    - Tests: Percentile calculations

14. **"Calculate the coefficient of variation for battery voltage by model"**
    - Tests: Complex statistical calculations

15. **"What's the interquartile range of mileage for vehicles with critical health status?"**
    - Tests: IQR calculations

## Category 4: Comparative Queries

16. **"Compare failure rates between Sentra and Leaf models for vehicles manufactured in 2023"**
    - Tests: Comparative analysis, filtering

17. **"Show me models where the failure rate is above the average failure rate across all models"**
    - Tests: Subqueries, comparative analysis

18. **"Which model has the highest failure rate among vehicles with mileage less than 50000?"**
    - Tests: Ranking with conditions, MAX with GROUP BY

19. **"Find models where battery failure rate is more than 2x the average battery failure rate"**
    - Tests: Subqueries, ratio comparisons

20. **"Compare average battery voltage between vehicles with battery failures and vehicles without failures"**
    - Tests: Conditional grouping, comparative analysis

## Category 5: Time-Based Complex Queries

21. **"What's the failure count by month for Sentra vehicles in 2023?"**
    - Tests: Date extraction, time-based grouping

22. **"Show me the trend of failures over the last 6 months grouped by model"**
    - Tests: Relative date filtering, time series

23. **"What's the average time between failures for each model?"**
    - Tests: Time difference calculations

24. **"Find vehicles that had failures in both Q1 and Q2 of 2023"**
    - Tests: Multiple time period filters, set operations

25. **"Calculate the failure rate by quarter for each model, showing only quarters with more than 5 failures"**
    - Tests: Quarter extraction, HAVING clause

## Category 6: Multi-Table/Join-Like Queries

26. **"Show me the correlation between battery voltage and battery degradation for each model"**
    - Tests: Correlation calculations, grouping

27. **"Find vehicles where battery_voltage is below average AND battery_degradation is above average for their model"**
    - Tests: Subqueries, model-specific averages

28. **"What's the ratio of battery failures to total failures for each model?"**
    - Tests: Conditional counting, ratio calculations

29. **"Show me models where the percentage of battery failures is above 50% of total failures"**
    - Tests: Percentage calculations, conditional aggregation

30. **"Calculate the average predicted claim probability for vehicles with actual failures vs vehicles without failures"**
    - Tests: Conditional grouping, comparative analysis

## Category 7: Complex Filtering with Multiple Criteria

31. **"Find all vehicles where (model is Sentra OR model is Leaf) AND (mileage between 0 and 50000) AND (battery_voltage < 12 OR battery_soc < 20)"**
    - Tests: Complex nested AND/OR logic

32. **"Show me vehicles with battery failures where mileage is in the top 10% AND battery degradation is in the bottom 10%"**
    - Tests: Percentile-based filtering

33. **"Find vehicles where failures_count is greater than the 75th percentile AND predicted_claim_probability is less than the 25th percentile"**
    - Tests: Percentile comparisons

34. **"List all vehicles where supplier_quality_score is below average AND defect_rate is above average"**
    - Tests: Multiple average comparisons

35. **"Find vehicles with battery failures where battery_voltage is more than 2 standard deviations below the mean"**
    - Tests: Statistical outlier detection

## Category 8: Ranking and Top-N Queries

36. **"What are the top 5 models by failure rate for vehicles manufactured in 2023?"**
    - Tests: Ranking, filtering, LIMIT

37. **"Show me the bottom 3 models by average battery voltage where failures_count is greater than 0"**
    - Tests: Reverse ranking, conditional filtering

38. **"Which 10 vehicles have the highest predicted claim probability among those with actual failures?"**
    - Tests: Conditional ranking, LIMIT

39. **"Find the top 3 failure modes (parts) by count for each model"**
    - Tests: Ranking within groups (window functions if supported)

40. **"Show me models ranked by failure rate, but only include models with at least 20 vehicles"**
    - Tests: Ranking with HAVING clause

## Category 9: Complex Calculations

41. **"Calculate the failure rate per 1000 miles for each model"**
    - Tests: Rate calculations, normalization

42. **"What's the weighted average of battery voltage by model, weighted by failures_count?"**
    - Tests: Weighted averages

43. **"Calculate the failure rate trend (slope) for each model over the last 12 months"**
    - Tests: Trend calculations, time series analysis

44. **"What's the compound failure rate for vehicles with multiple failure types?"**
    - Tests: Complex rate calculations

45. **"Show me the failure rate adjusted for mileage (failures per 10,000 miles) by model"**
    - Tests: Normalized rate calculations

## Category 10: Conditional Aggregations

46. **"What's the count of battery failures vs non-battery failures for each model?"**
    - Tests: Conditional counting, pivoting

47. **"Calculate the average battery voltage for vehicles with failures and separately for vehicles without failures, grouped by model"**
    - Tests: Conditional aggregation, grouping

48. **"Show me the failure rate for vehicles in different mileage buckets (0-25k, 25k-50k, 50k-75k, 75k+) by model"**
    - Tests: Bucketing, conditional aggregation

49. **"What's the percentage of vehicles with critical health status for each model?"**
    - Tests: Percentage calculations, conditional counting

50. **"Calculate the average predicted claim probability for vehicles that actually had failures vs those that didn't, by model"**
    - Tests: Conditional grouping, comparative analysis

## Category 11: Advanced Filtering

51. **"Find vehicles where the ratio of actual failures to predicted failures is greater than 1.5"**
    - Tests: Ratio calculations, comparative filtering

52. **"Show me models where battery failure rate increased by more than 20% from 2022 to 2023"**
    - Tests: Year-over-year comparisons, percentage change

53. **"Find vehicles with battery failures where battery_voltage is below the model-specific average"**
    - Tests: Model-specific subqueries

54. **"What's the failure rate for vehicles where all three conditions are true: low battery voltage (<12), high degradation (>10%), and low SOC (<20%)?"**
    - Tests: Multiple AND conditions, threshold filtering

55. **"Show me vehicles where the difference between predicted and actual failure probability is greater than 0.2"**
    - Tests: Difference calculations, threshold filtering

## Category 12: Complex Business Logic

56. **"Identify vehicles at high risk: where predicted_claim_probability > 0.6 AND (battery_voltage < 12 OR battery_degradation > 15%)"**
    - Tests: Business rule implementation, complex conditions

57. **"Find models where the failure rate for vehicles under 50k miles is higher than the failure rate for vehicles over 50k miles"**
    - Tests: Comparative analysis, conditional rates

58. **"Calculate the warranty cost risk score (failures_count * predicted_claim_probability) for each model"**
    - Tests: Calculated metrics, business logic

59. **"Show me vehicles that are outliers: where battery_voltage is more than 3 standard deviations from the model mean"**
    - Tests: Statistical outlier detection, model-specific calculations

60. **"Find models where supplier quality score is below 75 AND defect rate is above 0.2 AND failure rate is above 20%"**
    - Tests: Multiple business criteria, AND conditions

## Recommended Testing Sequence

### Start with Medium Complexity:
- Question #1: Multi-condition filtering
- Question #6: Complex aggregations with filters
- Question #16: Comparative queries

### Then Test High Complexity:
- Question #21: Time-based grouping
- Question #26: Correlation/statistical queries
- Question #36: Ranking with conditions

### Advanced Testing:
- Question #41: Complex calculations
- Question #46: Conditional aggregations
- Question #56: Business logic queries

## What to Verify

For each complex question, verify:
- ✅ SQL query is generated correctly
- ✅ All conditions are properly combined (AND/OR logic)
- ✅ Aggregations are calculated correctly
- ✅ Grouping works as expected
- ✅ Results are accurate
- ✅ Response is professional and analyst-friendly
- ✅ Industry context is provided when relevant
- ✅ Recommendations are specific and data-driven

