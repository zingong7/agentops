# Incident Review 2024

## Summary

Seven Sev-1 incidents in 2024, down from eleven in 2023. Total customer-facing
downtime was 3 hours 41 minutes. Four of the seven originated in `scan-ingest`.

## INC-2024-03: scan backlog during Easter peak

On 29 March the scan ingest consumer lag reached 46 minutes after a partition
rebalance stalled. Tracking data went stale but the API stayed available, so it
did not count against the availability SLO; it breached the freshness SLO
instead. Fixed by raising the consumer group's session timeout and adding a lag
alert at 5 minutes.

## INC-2024-07: routing solver OOM

On 12 July the routing solver was killed repeatedly by the OOM killer after a
customer booked a single shipment with 11,000 parcels. The solver had no cap on
problem size. Deliveries were unaffected because the previous route plan stayed
in force, but no new estimates were published for 4 hours. A hard cap of 2,500
parcels per solve was added.

## INC-2024-11: billing double-charge

On 2 November `billing-worker` reprocessed one night of shipments after a failed
deploy was rolled back mid-run. 1,340 customers were invoiced twice. The job had
no idempotency key. All charges were reversed within 48 hours. This was the only
2024 incident with direct financial impact.

## Follow-through

Of the 22 action items raised across the seven incidents, 19 were closed by year
end. The three still open all relate to `scan-ingest` partitioning.
