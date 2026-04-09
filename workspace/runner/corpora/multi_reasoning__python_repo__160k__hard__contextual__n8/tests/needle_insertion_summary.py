# Multi-Reasoning Retrieval Experiment: Needle Insertions

## Experiment Parameters
- **Experiment Type:** multi_reasoning
- **Discriminability:** hard (embedded in context)
- **Reference Clarity:** contextual (indirect description)
- **Number of Items:** 8

## Question
The system performs expired token cleanup at regular intervals. How many complete cleanup cycles could theoretically occur before all tokens expire naturally, assuming the various timing constraints across workers and APIs?

## Needles Inserted (8 items)

### target_001: Processing Throughput (workers/report_generator.md, line 91-92)
- **Value:** 250 rows per second
- **Context:** Embedded in report generator registration as performance metric comment
- **Why hard:** Not searchable as "250" or "throughput" - must read initialization context

### target_002: Database Connections (tests/conftest.md, line 50-51)  
- **Value:** 8 concurrent sessions maximum
- **Context:** Specified as MAX_CONCURRENT_SESSIONS in test configuration comments
- **Why hard:** Must understand connection pool purpose in cleanup operations

### target_003: Cleanup Batch Size (api/__init__.md, line 246)
- **Value:** 500 records per batch
- **Context:** Embedded in pagination docstring as "default batch size for cleanup operations"
- **Why hard:** Appears in pagination docs, not obvious it applies to cleanup

### target_004: Cleanup Cycle Interval (workers/sync_worker.md, line 25)
- **Value:** 3600 seconds (1 hour)
- **Context:** Comment about sync worker cleanup cycle timing
- **Why hard:** Mixed with other timing constants (STALE_LOCK_TIMEOUT), requires comprehension

### target_005: Token Expiration (api/auth.md, line 414)
- **Value:** 1 hour (TOKEN_EXPIRY_HOURS = 1)
- **Context:** Embedded in session creation return dictionary
- **Why hard:** Must infer from response value structure, not explicit constant definition

### target_006: Retry Backoff (workers/retry_handler.md, line 206)
- **Value:** 2x exponential multiplier
- **Context:** Specified in retry delay calculation logic comments
- **Why hard:** Requires understanding retry algorithm context

### target_007: Cleanup Task Schedule (tests/test_workers.md, line 176)
- **Value:** 1800 seconds (30 minutes)
- **Context:** Embedded in test configuration arguments for cleanup_expired_sessions
- **Why hard:** Mixed with other task parameters, requires reading surrounding test logic

### target_008: Cache TTL (api/rate_limiting.md, line 38)
- **Value:** 3600 seconds per token
- **Context:** Specified in rate limiter initialization
- **Why hard:** Implicit in cache design, not labeled as TTL explicitly

## Expected Answer
A correct answer should:
1. Calculate cleanup cycles: 3600 seconds / 1800 seconds = **2 complete cycles**
2. Show batch capacity: 8 sessions × 500 rows = 4000 total slots
3. Verify processing speed: 500 rows ÷ 250 rows/sec = 2 seconds (completes in-window)
4. Align sync worker cycle (3600s) with token expiry (3600s)
5. Reference all 8 values to demonstrate multi-reasoning requirement

## Answer Key Location
`/Users/mehrabmodi/Documents/projects/agent_retrieval_expt/workspace/judge/answer_keys/multi_reasoning__python_repo__160k__hard__contextual__n8.yaml`
