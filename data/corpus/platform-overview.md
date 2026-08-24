# Northwind Logistics Platform Overview

## Purpose

Northwind Logistics runs a parcel network across 14 countries. The platform is
the software layer that accepts shipment bookings, plans routes, and exposes
tracking to customers and partners.

## Services

The platform is six services:

- `booking-api`: accepts shipment bookings from the web app and partner APIs.
- `routing-engine`: assigns each parcel to a route and depot sequence.
- `tracking-api`: read-only scan history and delivery estimates.
- `scan-ingest`: consumes barcode scans from depot hardware.
- `billing-worker`: rates shipments and produces invoices nightly.
- `notify`: email and SMS delivery notifications.

All six are Python services on Kubernetes. `routing-engine` is the only one
written against a GPU node pool; it uses a solver that runs every 90 seconds.

## Ownership

The Platform team owns `booking-api`, `tracking-api` and `notify`. The Network
team owns `routing-engine` and `scan-ingest`. `billing-worker` is owned by
Finance Engineering, which sits outside the platform org.

## Traffic

Peak is 4,200 requests per second across the public APIs, reached in the two
weeks before Christmas. Baseline is roughly 900 requests per second. Scan
ingestion is steadier at about 1,100 events per second all year.
