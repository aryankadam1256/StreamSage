# Docker Interview Questions - Beginner

1. What is the difference between an image and a container?
2. Why do we use `docker-compose.yml` in microservices?
3. What does `FROM` do in a Dockerfile?
4. What does `WORKDIR` affect, host path or container path?
5. Difference between `RUN` and `CMD`.
6. What does `EXPOSE` do and what does it NOT do?
7. Why is `COPY requirements.txt` before `COPY . .` a best practice?
8. What is a bind mount?
9. What is a named volume?
10. Why can a container be running but still unhealthy?
11. How do containers discover each other in Docker Compose network?
12. Why should each service have its own Dockerfile?
13. What is the purpose of `.dockerignore`?
14. How do you check container logs quickly?
15. How do you stop all services in a compose project?

## Quick answer cues

- image = immutable blueprint, container = running instance.
- compose = orchestration, networking, env, mounts.
- RUN = build-time, CMD = run-time.
- WORKDIR is inside container FS.
- EXPOSE is metadata only; `ports` publishes to host.
