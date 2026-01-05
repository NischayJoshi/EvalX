# EvalX Scalability Architecture

## Services
1. API Service
   - Accepts submissions
   - Pushes jobs into a queue
   - Sends progress updates

2. PPT Evaluator Service
   - Processes PPT submissions independently

3. GitHub Evaluator Service
   - Processes repository submissions independently

## Queue System
- All submissions are placed into a queue
- Workers pull jobs from the queue
- Multiple workers allow parallel processing

## Real-Time Updates
- Users receive live updates:
  - "Submission received"
  - "Evaluation started"
  - "Running analysis"
  - "Evaluation complete"

## Caching
- If the same repository is submitted again
- Cached results are returned instantly

## Result
This design allows EvalX to scale horizontally
and handle large hackathon loads efficiently.
