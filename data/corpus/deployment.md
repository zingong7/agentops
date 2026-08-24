# Deployment Process

## Pipeline

Every merge to `main` builds a container image, runs the test suite, and deploys
to staging automatically. Production deploys are manual: an engineer promotes a
staging build through the deploy dashboard.

Production rollout is a canary. 5% of traffic for 10 minutes, then 50% for 10
minutes, then full. The canary aborts automatically if the 5xx rate on the new
pods exceeds 1% or p99 latency exceeds 400 ms.

## Freeze windows

There is a hard deploy freeze from 1 December to 5 January. Only changes with an
active Sev-1 or Sev-2 incident attached can be promoted during the freeze, and
they need sign-off from the on-call lead.

## Rollback

Rollback is a redeploy of the previous image tag from the dashboard and takes
about 90 seconds. Database migrations are not rolled back automatically;
migrations must be backward compatible with the previous release, which is
enforced by a check in CI that rejects destructive DDL in the same release as
the code that stops using a column.

## Environments

Staging runs against a scrubbed copy of production data refreshed weekly. There
is no long-lived development environment; engineers run services locally against
Docker Compose.
