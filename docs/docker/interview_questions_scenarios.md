# Docker Interview Questions - Scenario/Debug Round

1. Your API container is up, but healthcheck fails. How do you triage?
2. One service is healthy alone but unhealthy behind gateway. What do you check first?
3. A container loops with `exec format error`. List top 5 causes and fix order.
4. A Python package file in container is zero bytes. What likely happened?
5. ChromaDB says missing column on startup. How do you handle schema mismatch safely?
6. A model download keeps timing out during startup. How do you make startup resilient?
7. You moved project files to another drive and service mounts broke. Why and fix?
8. Frontend container starts but browser still shows old content. What do you inspect?
9. Docker Desktop works but compose says pipe/daemon unavailable intermittently. Recovery plan?
10. Your free-tier deployment sleeps and API times out. How do you redesign split architecture?

## Practical answer structure

For each scenario, answer in this order:
1. Symptom confirmation command.
2. Most probable root causes.
3. Minimal safe fix.
4. Verification command.
5. Long-term prevention.
