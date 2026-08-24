# Data Retention and Privacy

## Retention periods

- Scan events: 24 months, then aggregated to daily counts and the raw rows dropped.
- Shipment records: 7 years, driven by tax rules in the operating countries.
- Recipient contact details: 90 days after final delivery scan.
- Application logs: 30 days in hot storage, 12 months in cold storage.

## Deletion requests

Deletion requests arrive through the support tool and are processed nightly by
`privacy-worker`. Shipment records cannot be deleted before the 7-year point
because of the tax retention obligation, so the worker redacts recipient name,
address and contact fields in place and leaves the shipment row intact.

The service level for processing a request is 14 days; the nightly job normally
clears the queue within 24 hours.

## Access

Production data access requires a time-boxed grant of at most 8 hours, approved
by a team lead. All queries against production databases are logged to a
separate audit cluster that engineers cannot write to.
