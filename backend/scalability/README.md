# Scalability Upgrades – EvalX

## Problem
EvalX currently evaluates PPTs and GitHub repositories synchronously.
When many teams submit at once, later submissions must wait, causing delays.

## Goal
Enable EvalX to handle 1000+ concurrent submissions without performance degradation.

## Proposed Architecture
- Separate evaluators into independent services
- Use a queue-based processing system
- Process submissions asynchronously with workers
- Send live progress updates to users
- Cache results to avoid recomputation

## What This Folder Contains
This folder demonstrates a **scalable design and working simulation**
of how EvalX can process submissions concurrently.

The implementation is intentionally lightweight and hackathon-friendly.
