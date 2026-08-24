# Tracking API Reference

## Service level objectives

The tracking API is held to three SLOs, measured over a rolling 28-day window:

- Availability: 99.9% of requests return a non-5xx response.
- Latency: p99 response time under 250 ms, p50 under 40 ms.
- Freshness: 95% of depot scans are queryable within 30 seconds of ingest.

Breaching any SLO for two consecutive windows triggers a feature freeze on the
owning team until the error budget recovers.

## Endpoints

`GET /v1/parcels/{tracking_number}` returns the current status and the full scan
history. `GET /v1/parcels/{tracking_number}/eta` returns the predicted delivery
window. Both require an API key passed in the `X-Northwind-Key` header.

Partner keys are rate limited to 600 requests per minute. Internal keys are not
rate limited but are restricted by network policy to the cluster.

## Caching

Scan history is cached in Redis for 20 seconds. ETA responses are cached for 5
minutes because the routing solver only republishes estimates every 90 seconds.
Cache is keyed on tracking number alone; there is no per-caller variation.

## Known limitations

The API does not expose depot names for parcels in transit through partner
carriers, only the country. This is a contractual restriction, not a technical
one, and it applies to about 8% of international shipments.
