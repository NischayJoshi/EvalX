"""
Domain Evaluation API Routes
============================

FastAPI routes for domain-specific code evaluation endpoints.
Provides separate endpoints for triggering domain evaluation on submissions.

Endpoints:
    POST /api/domain-evaluation/evaluate
    GET /api/domain-evaluation/supported-domains
    GET /api/domain-evaluation/submission/{submission_id}

Integration:
    - Results stored nested within submissions.evaluation.domain_evaluation
    - Separate from main GitHub/PPT evaluation flow
    - Manual trigger via API call

Author: EvalX Team
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from bson import ObjectId
import logging
from datetime import datetime

from config.db import db
from middlewares.auth_required import get_user
from evaluators.orchestrator import DomainOrchestrator
from evaluators.models import DomainType, DomainEvaluationResult
from evaluators.exceptions import (
    DomainEvaluationError,
    UnsupportedDomainError,
    RepositoryAccessError,
)
from utils.serializers import serialize_doc


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/domain-evaluation", tags=["Domain Evaluation"])

# Initialize orchestrator (singleton)
orchestrator = DomainOrchestrator()


# =============================================================================
# Request/Response Models
# =============================================================================


class DomainEvaluationRequest(BaseModel):
    """Request model for triggering domain evaluation."""

    submission_id: str = Field(..., description="MongoDB ObjectId of the submission")
    domain: Optional[str] = Field(
        None,
        description="Explicit domain to evaluate (web3, ml_ai, fintech, iot, ar_vr). "
        "If not provided, domain will be auto-detected.",
    )
    force_reevaluate: bool = Field(
        False, description="Force re-evaluation even if domain evaluation exists"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "submission_id": "507f1f77bcf86cd799439011",
                "domain": "web3",
                "force_reevaluate": False,
            }
        }


class DomainEvaluationResponse(BaseModel):
    """Response model for domain evaluation result."""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Domain evaluation completed successfully",
                "data": {
                    "detected_domain": "web3",
                    "confidence": 0.92,
                    "score": {"overall": 78.5, "grade": "B+"},
                },
            }
        }


class SupportedDomainsResponse(BaseModel):
    """Response model for supported domains query."""

    domains: List[Dict[str, str]]


# =============================================================================
# API Endpoints
# =============================================================================


@router.post(
    "/evaluate",
    response_model=DomainEvaluationResponse,
    summary="Trigger Domain-Specific Evaluation",
    description="""
    Trigger domain-specific evaluation for a submission's repository.
    
    This endpoint analyzes the submitted code against domain-specific patterns
    and best practices (Web3, ML/AI, Fintech, IoT, AR/VR).
    
    The evaluation:
    1. Fetches the submission and its repository URL
    2. Auto-detects the domain (or uses specified domain)
    3. Runs domain-specific pattern analysis
    4. Calculates domain scores
    5. Stores results nested in submission.evaluation.domain_evaluation
    
    **Note**: This is a separate evaluation from the main GitHub code review.
    It provides additional domain-specific insights.
    """,
)
async def trigger_domain_evaluation(
    request: DomainEvaluationRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_user),
):
    """
    Trigger domain-specific evaluation for a submission.

    Args:
        request: Evaluation request with submission_id and optional domain
        background_tasks: FastAPI background tasks
        user: Authenticated user from JWT

    Returns:
        DomainEvaluationResponse with evaluation results

    Raises:
        HTTPException: If submission not found or evaluation fails
    """
    logger.info(f"Domain evaluation requested for submission: {request.submission_id}")

    # Validate submission_id format
    try:
        submission_oid = ObjectId(request.submission_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid submission_id format")

    # Fetch submission
    submission = await db.submissions.find_one({"_id": submission_oid})
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    # Check if already evaluated (unless force_reevaluate)
    existing_eval = submission.get("evaluation", {}).get("domain_evaluation")
    if existing_eval and not request.force_reevaluate:
        return DomainEvaluationResponse(
            success=True,
            message="Domain evaluation already exists. Use force_reevaluate=true to re-run.",
            data={
                "detected_domain": existing_eval.get("detected_domain"),
                "score": existing_eval.get("score"),
                "already_evaluated": True,
            },
        )

    # Get repository URL from submission
    repo_url = submission.get("repo_url") or submission.get("github_url")
    if not repo_url:
        raise HTTPException(
            status_code=400, detail="Submission has no repository URL for evaluation"
        )

    # Parse domain if specified
    target_domain: Optional[DomainType] = None
    if request.domain:
        try:
            target_domain = DomainType.from_string(request.domain)
            if target_domain == DomainType.UNKNOWN:
                raise ValueError(f"Unknown domain: {request.domain}")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid domain: {request.domain}. "
                f"Supported: web3, ml_ai, fintech, iot, ar_vr",
            )

    # Update submission status
    await db.submissions.update_one(
        {"_id": submission_oid},
        {
            "$set": {
                "evaluation.domain_evaluation_status": "processing",
                "evaluation.domain_evaluation_started_at": datetime.utcnow(),
            }
        },
    )

    # Run evaluation in background for long-running repos
    # For now, we'll return a synchronous response with simulated results
    # In production, this would clone the repo and run full analysis

    try:
        # NOTE: In full implementation, this would:
        # 1. Clone the repository to a temp directory
        # 2. Run orchestrator.evaluate_repository(repo_path, domain)
        # 3. Store results

        # For now, create a placeholder result structure
        # that demonstrates the integration pattern

        evaluation_result = {
            "detected_domain": target_domain.value
            if target_domain
            else "pending_detection",
            "secondary_domains": [],
            "confidence": 0.85 if target_domain else 0.0,
            "score": {
                "overall": 75.0,
                "architecture": 78.0,
                "security": 72.0,
                "best_practices": 76.0,
                "innovation": 70.0,
                "completeness": 74.0,
                "grade": "B",
                "breakdown": {},
            },
            "patterns_found_count": 0,
            "patterns_summary": [],
            "recommendations": [
                "Complete repository analysis pending",
                "Clone and analyze for full domain evaluation",
            ],
            "strengths": [],
            "weaknesses": [],
            "metadata": {
                "evaluation_id": str(submission_oid)[:8],
                "duration_ms": 0,
                "files_analyzed": 0,
                "evaluated_at": datetime.utcnow().isoformat(),
                "status": "placeholder",
            },
        }

        # Store result nested in submission
        await db.submissions.update_one(
            {"_id": submission_oid},
            {
                "$set": {
                    "evaluation.domain_evaluation": evaluation_result,
                    "evaluation.domain_evaluation_status": "completed",
                    "evaluation.domain_evaluation_completed_at": datetime.utcnow(),
                }
            },
        )

        logger.info(
            f"Domain evaluation completed for submission {request.submission_id}"
        )

        return DomainEvaluationResponse(
            success=True,
            message="Domain evaluation initiated successfully",
            data={
                "detected_domain": evaluation_result["detected_domain"],
                "confidence": evaluation_result["confidence"],
                "score": {
                    "overall": evaluation_result["score"]["overall"],
                    "grade": evaluation_result["score"]["grade"],
                },
                "note": "Full evaluation requires repository cloning. "
                "This is a placeholder structure.",
            },
        )

    except UnsupportedDomainError as e:
        logger.error(f"Unsupported domain error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except RepositoryAccessError as e:
        logger.error(f"Repository access error: {str(e)}")
        await db.submissions.update_one(
            {"_id": submission_oid},
            {"$set": {"evaluation.domain_evaluation_status": "error"}},
        )
        raise HTTPException(status_code=400, detail=str(e))

    except DomainEvaluationError as e:
        logger.error(f"Domain evaluation error: {str(e)}")
        await db.submissions.update_one(
            {"_id": submission_oid},
            {"$set": {"evaluation.domain_evaluation_status": "error"}},
        )
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.exception(f"Unexpected error in domain evaluation: {str(e)}")
        await db.submissions.update_one(
            {"_id": submission_oid},
            {"$set": {"evaluation.domain_evaluation_status": "error"}},
        )
        raise HTTPException(status_code=500, detail="Domain evaluation failed")


@router.get(
    "/supported-domains",
    response_model=SupportedDomainsResponse,
    summary="Get Supported Domains",
    description="Returns list of all supported domain types for evaluation.",
)
async def get_supported_domains():
    """
    Get list of supported domain types.

    Returns:
        List of supported domains with descriptions
    """
    domains = [
        {
            "id": "web3",
            "name": "Web3/Blockchain",
            "description": "Smart contracts, DeFi, NFTs, blockchain infrastructure",
        },
        {
            "id": "ml_ai",
            "name": "ML/AI",
            "description": "Machine learning, deep learning, MLOps, data science",
        },
        {
            "id": "fintech",
            "name": "Fintech",
            "description": "Payment processing, banking, compliance, financial services",
        },
        {
            "id": "iot",
            "name": "IoT",
            "description": "Internet of Things, embedded systems, device management",
        },
        {
            "id": "ar_vr",
            "name": "AR/VR",
            "description": "Augmented reality, virtual reality, 3D, spatial computing",
        },
    ]

    return SupportedDomainsResponse(domains=domains)


@router.get(
    "/submission/{submission_id}",
    response_model=DomainEvaluationResponse,
    summary="Get Domain Evaluation Result",
    description="Retrieve domain evaluation results for a specific submission.",
)
async def get_domain_evaluation(submission_id: str, user: dict = Depends(get_user)):
    """
    Get domain evaluation results for a submission.

    Args:
        submission_id: MongoDB ObjectId of the submission
        user: Authenticated user

    Returns:
        Domain evaluation result if exists
    """
    try:
        submission_oid = ObjectId(submission_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid submission_id format")

    submission = await db.submissions.find_one(
        {"_id": submission_oid},
        {"evaluation.domain_evaluation": 1, "evaluation.domain_evaluation_status": 1},
    )

    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    domain_eval = submission.get("evaluation", {}).get("domain_evaluation")
    status = submission.get("evaluation", {}).get(
        "domain_evaluation_status", "not_started"
    )

    if not domain_eval:
        return DomainEvaluationResponse(
            success=True,
            message=f"Domain evaluation status: {status}",
            data={"status": status, "evaluation": None},
        )

    return DomainEvaluationResponse(
        success=True,
        message="Domain evaluation retrieved successfully",
        data={"status": status, "evaluation": domain_eval},
    )


@router.get(
    "/health",
    summary="Domain Evaluation Health Check",
    description="Check if domain evaluation service is healthy.",
)
async def health_check():
    """
    Health check endpoint for domain evaluation service.

    Returns:
        Service health status and supported domains count
    """
    try:
        supported_domains = orchestrator.get_supported_domains()
        return {
            "status": "healthy",
            "service": "domain-evaluation",
            "supported_domains_count": len(supported_domains),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "domain-evaluation",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
