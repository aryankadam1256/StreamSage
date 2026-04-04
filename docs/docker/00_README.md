# StreamSage Docker Docs Pack

This folder contains a complete Docker learning + implementation record for this project.

## Files

- `01_current_status.md`: exact current status and pending items
- `02_commands_run_and_why.md`: command-by-command history with purpose
- `03_internal_working_theory.md`: what happens internally during build/run/networking
- `04_deployment_paths_and_free_options.md`: deployment options after Dockerization
- `05_troubleshooting_log.md`: major issues found and how they were fixed
- `06_deployment_implementation_plan.md`: full rollout plan from image push to split deployment
- `07_env_matrix_vercel_render_hf.md`: environment variable matrix by platform
- `08_registry_pull_flow.md`: how to pull all microservice images together using compose
- `09_operations_playbook.md`: daily Docker runbook (start, health, smoke, rebuild, logs)
- `interview_questions_beginner.md`: beginner interview prep questions
- `interview_questions_intermediate.md`: intermediate interview prep questions
- `interview_questions_scenarios.md`: scenario/debug interview questions

## Automation scripts

- `scripts/deploy/docker_stack.ps1`: single entrypoint for routine Docker operations
- `scripts/deploy/verify_stack_health.ps1`: health validation for gateway and all backend services
- `scripts/deploy/run_smoke_tests.ps1`: end-to-end smoke tests for discover + Oracle sync/stream paths

## Important workspace note

Project source of truth is now on:

`D:\StreamSage`

If VS Code explorer is opened on `C:\Users\mailp\StreamSage`, open the D path as workspace to see all files immediately.
