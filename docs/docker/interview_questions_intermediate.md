# Docker Interview Questions - Intermediate

1. Explain Docker layer caching with a real Python service example.
2. What are tradeoffs of single-stage vs multi-stage builds?
3. How does Docker bridge networking work in Compose?
4. Why can host `localhost` fail for inter-container communication?
5. What does `depends_on` guarantee and what does it NOT guarantee?
6. How would you reduce image size in a Python service?
7. How do you debug a service that keeps restarting with exit code 0?
8. Why would `--no-cache` be needed during rebuild?
9. Explain bind mount vs named volume for ML model artifacts.
10. How do you persist large model cache between container recreations?
11. What causes "works on local, fails in container" issues most often?
12. How do you make healthchecks meaningful for slow-start services?
13. Why does build context size matter and how do you control it?
14. How do you safely migrate Docker data root on Windows?
15. Why can dependency files become corrupted during low-disk events?

## Discussion points

- Cache key invalidation by file changes.
- Multi-stage isolates runtime from build toolchain.
- Service DNS via compose network.
- Mount strategy affects portability and ops.
- Healthchecks should reflect readiness, not just process liveness.
