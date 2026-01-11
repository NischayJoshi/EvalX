"""
Domain Pattern Constants
========================

Comprehensive regex patterns and detection rules for each supported domain.
These patterns are carefully crafted to identify domain-specific code constructs,
best practices, security vulnerabilities, and architectural patterns.

Pattern Structure:
    Each pattern dictionary contains:
    - regex: The detection regular expression
    - category: Pattern classification (security, architecture, best_practice, etc.)
    - severity: Impact level (info, low, medium, high, critical)
    - description: Human-readable explanation
    - weight: Scoring weight multiplier
    - confidence: Default detection confidence (0.0-1.0)

Author: EvalX Team
"""

from typing import Dict, Any

# =============================================================================
# WEB3 / BLOCKCHAIN PATTERNS (Priority Domain)
# =============================================================================

WEB3_PATTERNS: Dict[str, Dict[str, Any]] = {
    # Smart Contract Security Patterns
    "reentrancy_guard": {
        "regex": r"(nonReentrant|ReentrancyGuard|_notEntered|reentrancy\s*lock)",
        "category": "security",
        "severity": "high",
        "description": "Reentrancy protection mechanism to prevent recursive call attacks",
        "weight": 2.0,
        "confidence": 0.95
    },
    "access_control": {
        "regex": r"(onlyOwner|onlyAdmin|AccessControl|Ownable|hasRole\s*\(|grantRole|revokeRole)",
        "category": "security",
        "severity": "high",
        "description": "Access control pattern for privileged operations",
        "weight": 1.8,
        "confidence": 0.92
    },
    "safe_math": {
        "regex": r"(SafeMath|unchecked\s*{|overflow|underflow|\.add\(|\.sub\(|\.mul\(|\.div\()",
        "category": "security",
        "severity": "medium",
        "description": "Safe arithmetic operations to prevent overflow/underflow",
        "weight": 1.5,
        "confidence": 0.88
    },
    "pausable_pattern": {
        "regex": r"(Pausable|whenNotPaused|whenPaused|_pause\(\)|_unpause\(\)|paused\s*\(\))",
        "category": "security",
        "severity": "medium",
        "description": "Circuit breaker pattern for emergency stops",
        "weight": 1.4,
        "confidence": 0.90
    },
    "upgradeable_proxy": {
        "regex": r"(Proxy|Upgradeable|TransparentProxy|UUPS|initializer|__gap|_authorizeUpgrade)",
        "category": "architecture",
        "severity": "info",
        "description": "Upgradeable smart contract pattern",
        "weight": 1.3,
        "confidence": 0.85
    },
    
    # DeFi Patterns
    "erc20_implementation": {
        "regex": r"(ERC20|IERC20|transfer\(|transferFrom\(|approve\(|allowance\(|balanceOf\()",
        "category": "architecture",
        "severity": "info",
        "description": "ERC-20 token standard implementation",
        "weight": 1.2,
        "confidence": 0.93
    },
    "erc721_nft": {
        "regex": r"(ERC721|IERC721|tokenURI|ownerOf|safeTransferFrom|_mint\(|_safeMint\()",
        "category": "architecture",
        "severity": "info",
        "description": "ERC-721 NFT standard implementation",
        "weight": 1.2,
        "confidence": 0.93
    },
    "erc1155_multitoken": {
        "regex": r"(ERC1155|IERC1155|balanceOfBatch|safeBatchTransferFrom|uri\()",
        "category": "architecture",
        "severity": "info",
        "description": "ERC-1155 multi-token standard implementation",
        "weight": 1.2,
        "confidence": 0.91
    },
    "liquidity_pool": {
        "regex": r"(LiquidityPool|addLiquidity|removeLiquidity|swap\(|getReserves|k\s*=\s*x\s*\*\s*y)",
        "category": "architecture",
        "severity": "info",
        "description": "Automated Market Maker (AMM) liquidity pool pattern",
        "weight": 1.5,
        "confidence": 0.87
    },
    "flash_loan": {
        "regex": r"(flashLoan|FlashLoan|executeOperation|FLASHLOAN_PREMIUM|flashBorrower)",
        "category": "architecture",
        "severity": "medium",
        "description": "Flash loan functionality implementation",
        "weight": 1.4,
        "confidence": 0.89
    },
    "staking_mechanism": {
        "regex": r"(stake\(|unstake\(|Staking|rewardRate|earned\(|getReward|stakingToken)",
        "category": "architecture",
        "severity": "info",
        "description": "Token staking and rewards mechanism",
        "weight": 1.3,
        "confidence": 0.88
    },
    
    # Blockchain Infrastructure
    "event_emission": {
        "regex": r"(emit\s+\w+\(|event\s+\w+\(|indexed\s+)",
        "category": "best_practice",
        "severity": "info",
        "description": "Event emission for off-chain indexing and tracking",
        "weight": 1.0,
        "confidence": 0.95
    },
    "oracle_integration": {
        "regex": r"(Oracle|Chainlink|AggregatorV3|latestRoundData|priceFeed|getPrice\()",
        "category": "architecture",
        "severity": "info",
        "description": "Oracle integration for external data feeds",
        "weight": 1.4,
        "confidence": 0.86
    },
    "gas_optimization": {
        "regex": r"(calldata|memory|storage|immutable\s+|constant\s+|unchecked\s*{)",
        "category": "best_practice",
        "severity": "info",
        "description": "Gas optimization techniques",
        "weight": 1.1,
        "confidence": 0.80
    },
    "signature_verification": {
        "regex": r"(ecrecover|ECDSA|signTypedData|EIP712|verifySignature|_hashTypedDataV4)",
        "category": "security",
        "severity": "high",
        "description": "Cryptographic signature verification",
        "weight": 1.6,
        "confidence": 0.90
    },
    "timelock_pattern": {
        "regex": r"(Timelock|TimelockController|delay|queueTransaction|executeTransaction|eta)",
        "category": "security",
        "severity": "medium",
        "description": "Timelock pattern for governance actions",
        "weight": 1.4,
        "confidence": 0.88
    },
}

# =============================================================================
# ML/AI PATTERNS (Priority Domain)
# =============================================================================

ML_AI_PATTERNS: Dict[str, Dict[str, Any]] = {
    # Framework Detection
    "tensorflow_usage": {
        "regex": r"(import\s+tensorflow|from\s+tensorflow|tf\.|keras\.|tf\.keras)",
        "category": "framework",
        "severity": "info",
        "description": "TensorFlow/Keras framework usage",
        "weight": 1.2,
        "confidence": 0.95
    },
    "pytorch_usage": {
        "regex": r"(import\s+torch|from\s+torch|torch\.|nn\.Module|torch\.nn)",
        "category": "framework",
        "severity": "info",
        "description": "PyTorch framework usage",
        "weight": 1.2,
        "confidence": 0.95
    },
    "sklearn_usage": {
        "regex": r"(from\s+sklearn|import\s+sklearn|sklearn\.|fit\(|predict\(|transform\()",
        "category": "framework",
        "severity": "info",
        "description": "Scikit-learn library usage",
        "weight": 1.1,
        "confidence": 0.93
    },
    "huggingface_transformers": {
        "regex": r"(from\s+transformers|AutoModel|AutoTokenizer|pipeline\(|PreTrainedModel)",
        "category": "framework",
        "severity": "info",
        "description": "Hugging Face Transformers library usage",
        "weight": 1.4,
        "confidence": 0.94
    },
    
    # Model Architecture Patterns
    "neural_network_definition": {
        "regex": r"(class\s+\w+\(nn\.Module\)|Sequential\(|Dense\(|Linear\(|Conv2d\(|LSTM\(|GRU\()",
        "category": "architecture",
        "severity": "info",
        "description": "Neural network architecture definition",
        "weight": 1.5,
        "confidence": 0.90
    },
    "attention_mechanism": {
        "regex": r"(Attention|MultiHeadAttention|self_attention|cross_attention|softmax\(.*\/.*sqrt)",
        "category": "architecture",
        "severity": "info",
        "description": "Attention mechanism implementation",
        "weight": 1.6,
        "confidence": 0.88
    },
    "transformer_architecture": {
        "regex": r"(Transformer|TransformerEncoder|TransformerDecoder|positional_encoding|LayerNorm)",
        "category": "architecture",
        "severity": "info",
        "description": "Transformer architecture components",
        "weight": 1.7,
        "confidence": 0.89
    },
    "cnn_architecture": {
        "regex": r"(Conv2d|Conv1d|MaxPool|AvgPool|BatchNorm|Dropout|Flatten)",
        "category": "architecture",
        "severity": "info",
        "description": "Convolutional Neural Network components",
        "weight": 1.3,
        "confidence": 0.91
    },
    "rnn_architecture": {
        "regex": r"(LSTM|GRU|RNN|hidden_state|cell_state|bidirectional|num_layers)",
        "category": "architecture",
        "severity": "info",
        "description": "Recurrent Neural Network components",
        "weight": 1.3,
        "confidence": 0.90
    },
    
    # Training Pipeline Patterns
    "training_loop": {
        "regex": r"(for\s+epoch|optimizer\.step|loss\.backward|model\.train\(\)|train_loader)",
        "category": "best_practice",
        "severity": "info",
        "description": "Model training loop implementation",
        "weight": 1.4,
        "confidence": 0.88
    },
    "data_augmentation": {
        "regex": r"(RandomCrop|RandomFlip|RandomRotation|Augment|transforms\.Compose|albumentations)",
        "category": "best_practice",
        "severity": "info",
        "description": "Data augmentation techniques",
        "weight": 1.2,
        "confidence": 0.87
    },
    "learning_rate_scheduler": {
        "regex": r"(lr_scheduler|StepLR|CosineAnnealing|ReduceLROnPlateau|WarmupScheduler|OneCycleLR)",
        "category": "best_practice",
        "severity": "info",
        "description": "Learning rate scheduling strategy",
        "weight": 1.3,
        "confidence": 0.89
    },
    "early_stopping": {
        "regex": r"(EarlyStopping|patience|best_loss|early_stop|val_loss.*<.*best)",
        "category": "best_practice",
        "severity": "info",
        "description": "Early stopping regularization",
        "weight": 1.2,
        "confidence": 0.85
    },
    "gradient_clipping": {
        "regex": r"(clip_grad_norm|clip_grad_value|max_norm|gradient_clip|torch\.nn\.utils\.clip)",
        "category": "best_practice",
        "severity": "info",
        "description": "Gradient clipping for training stability",
        "weight": 1.1,
        "confidence": 0.90
    },
    
    # MLOps Patterns
    "model_checkpointing": {
        "regex": r"(save_checkpoint|load_checkpoint|torch\.save|torch\.load|model\.save|ModelCheckpoint)",
        "category": "best_practice",
        "severity": "info",
        "description": "Model checkpointing for training recovery",
        "weight": 1.3,
        "confidence": 0.91
    },
    "experiment_tracking": {
        "regex": r"(mlflow|wandb|tensorboard|SummaryWriter|log_metric|log_param|neptune)",
        "category": "best_practice",
        "severity": "info",
        "description": "Experiment tracking and logging",
        "weight": 1.4,
        "confidence": 0.92
    },
    "model_versioning": {
        "regex": r"(dvc|MLflow\.register|model_registry|version.*model|ModelVersion)",
        "category": "best_practice",
        "severity": "info",
        "description": "Model versioning and registry",
        "weight": 1.3,
        "confidence": 0.85
    },
    "hyperparameter_tuning": {
        "regex": r"(Optuna|hyperopt|GridSearch|RandomizedSearch|Ray\[tune\]|hparam|hyperparameter)",
        "category": "best_practice",
        "severity": "info",
        "description": "Hyperparameter optimization",
        "weight": 1.4,
        "confidence": 0.88
    },
    
    # Inference Patterns
    "model_inference": {
        "regex": r"(model\.eval\(\)|torch\.no_grad|inference|predict_proba|model\.predict)",
        "category": "architecture",
        "severity": "info",
        "description": "Model inference implementation",
        "weight": 1.2,
        "confidence": 0.89
    },
    "model_quantization": {
        "regex": r"(quantize|int8|float16|mixed_precision|amp|torch\.quantization|TensorRT)",
        "category": "best_practice",
        "severity": "info",
        "description": "Model quantization for efficiency",
        "weight": 1.3,
        "confidence": 0.86
    },
    "onnx_export": {
        "regex": r"(onnx|torch\.onnx|onnxruntime|export.*onnx|ONNX)",
        "category": "best_practice",
        "severity": "info",
        "description": "ONNX model export for portability",
        "weight": 1.2,
        "confidence": 0.90
    },
}

# =============================================================================
# FINTECH PATTERNS (Priority Domain)
# =============================================================================

FINTECH_PATTERNS: Dict[str, Dict[str, Any]] = {
    # Payment Processing
    "payment_gateway": {
        "regex": r"(Stripe|PayPal|Braintree|Adyen|Square|payment_intent|charge\(|refund\()",
        "category": "architecture",
        "severity": "info",
        "description": "Payment gateway integration",
        "weight": 1.5,
        "confidence": 0.93
    },
    "payment_processing": {
        "regex": r"(process_payment|payment_method|card_number|cvv|expiry|billing_address)",
        "category": "architecture",
        "severity": "medium",
        "description": "Payment processing logic",
        "weight": 1.4,
        "confidence": 0.88
    },
    "recurring_billing": {
        "regex": r"(subscription|recurring|billing_cycle|invoice|auto_renew|plan_id)",
        "category": "architecture",
        "severity": "info",
        "description": "Subscription and recurring billing",
        "weight": 1.3,
        "confidence": 0.87
    },
    
    # Security & Compliance
    "pci_compliance": {
        "regex": r"(PCI|tokenize|vault|card_token|sensitive_data|encrypt.*card|mask.*number)",
        "category": "security",
        "severity": "critical",
        "description": "PCI-DSS compliance patterns",
        "weight": 2.0,
        "confidence": 0.85
    },
    "kyc_aml": {
        "regex": r"(KYC|AML|identity_verification|document_verification|sanctions|watchlist|PEP)",
        "category": "security",
        "severity": "high",
        "description": "KYC/AML compliance implementation",
        "weight": 1.8,
        "confidence": 0.86
    },
    "fraud_detection": {
        "regex": r"(fraud|risk_score|anomaly|suspicious|velocity_check|device_fingerprint|behavioral)",
        "category": "security",
        "severity": "high",
        "description": "Fraud detection mechanisms",
        "weight": 1.7,
        "confidence": 0.84
    },
    "transaction_security": {
        "regex": r"(idempotency|idempotent|transaction_id|double_spend|replay_attack|nonce)",
        "category": "security",
        "severity": "high",
        "description": "Transaction security patterns",
        "weight": 1.6,
        "confidence": 0.88
    },
    "encryption_at_rest": {
        "regex": r"(encrypt.*rest|AES|KMS|key_management|HSM|envelope_encryption|data_key)",
        "category": "security",
        "severity": "high",
        "description": "Data encryption at rest",
        "weight": 1.7,
        "confidence": 0.87
    },
    
    # Banking Patterns
    "account_management": {
        "regex": r"(account_balance|ledger|debit|credit|transfer|withdrawal|deposit)",
        "category": "architecture",
        "severity": "info",
        "description": "Banking account management",
        "weight": 1.4,
        "confidence": 0.89
    },
    "double_entry_bookkeeping": {
        "regex": r"(double_entry|ledger_entry|debit.*credit|journal_entry|contra_account)",
        "category": "architecture",
        "severity": "info",
        "description": "Double-entry bookkeeping system",
        "weight": 1.5,
        "confidence": 0.82
    },
    "transaction_ledger": {
        "regex": r"(transaction_log|audit_trail|immutable.*log|event_sourcing|CQRS)",
        "category": "architecture",
        "severity": "info",
        "description": "Transaction ledger and audit trail",
        "weight": 1.4,
        "confidence": 0.85
    },
    "interest_calculation": {
        "regex": r"(interest_rate|APR|APY|compound_interest|accrued_interest|amortization)",
        "category": "architecture",
        "severity": "info",
        "description": "Interest calculation logic",
        "weight": 1.2,
        "confidence": 0.86
    },
    
    # Open Banking
    "open_banking_api": {
        "regex": r"(OpenBanking|PSD2|AISP|PISP|account_information|payment_initiation|consent)",
        "category": "architecture",
        "severity": "info",
        "description": "Open Banking API integration",
        "weight": 1.4,
        "confidence": 0.84
    },
    "plaid_integration": {
        "regex": r"(Plaid|plaid_client|link_token|access_token|institution_id|accounts/get)",
        "category": "architecture",
        "severity": "info",
        "description": "Plaid financial data integration",
        "weight": 1.3,
        "confidence": 0.91
    },
    
    # Regulatory
    "regulatory_reporting": {
        "regex": r"(regulatory.*report|compliance.*report|SAR|CTR|Form.*8300|FinCEN)",
        "category": "security",
        "severity": "medium",
        "description": "Regulatory reporting implementation",
        "weight": 1.5,
        "confidence": 0.80
    },
    "audit_logging": {
        "regex": r"(audit_log|audit.*trail|who.*what.*when|immutable.*record|compliance.*log)",
        "category": "security",
        "severity": "medium",
        "description": "Comprehensive audit logging",
        "weight": 1.4,
        "confidence": 0.86
    },
}

# =============================================================================
# IOT PATTERNS (Lighter Implementation)
# =============================================================================

IOT_PATTERNS: Dict[str, Dict[str, Any]] = {
    # Communication Protocols
    "mqtt_protocol": {
        "regex": r"(MQTT|mqtt|paho|publish\(|subscribe\(|broker|QoS|topic)",
        "category": "architecture",
        "severity": "info",
        "description": "MQTT messaging protocol usage",
        "weight": 1.4,
        "confidence": 0.92
    },
    "coap_protocol": {
        "regex": r"(CoAP|coap|constrained.*application|aiocoap|coapthon)",
        "category": "architecture",
        "severity": "info",
        "description": "CoAP protocol for constrained devices",
        "weight": 1.3,
        "confidence": 0.88
    },
    "websocket_iot": {
        "regex": r"(WebSocket|ws://|wss://|socket\.io|real.*time.*device)",
        "category": "architecture",
        "severity": "info",
        "description": "WebSocket for real-time device communication",
        "weight": 1.2,
        "confidence": 0.85
    },
    
    # Device Management
    "device_provisioning": {
        "regex": r"(device_id|provision|registration|onboard|device_twin|shadow)",
        "category": "architecture",
        "severity": "info",
        "description": "Device provisioning and registration",
        "weight": 1.3,
        "confidence": 0.86
    },
    "ota_update": {
        "regex": r"(OTA|firmware.*update|over.*the.*air|bootloader|flash|upgrade)",
        "category": "architecture",
        "severity": "medium",
        "description": "Over-the-Air update mechanism",
        "weight": 1.5,
        "confidence": 0.84
    },
    "device_telemetry": {
        "regex": r"(telemetry|sensor.*data|metrics|heartbeat|device.*status|health.*check)",
        "category": "architecture",
        "severity": "info",
        "description": "Device telemetry and monitoring",
        "weight": 1.3,
        "confidence": 0.88
    },
    
    # Sensor Integration
    "sensor_reading": {
        "regex": r"(read_sensor|temperature|humidity|pressure|accelerometer|gyroscope|GPIO)",
        "category": "architecture",
        "severity": "info",
        "description": "Sensor data reading implementation",
        "weight": 1.2,
        "confidence": 0.87
    },
    "actuator_control": {
        "regex": r"(actuator|motor|servo|relay|PWM|digital_write|analog_write)",
        "category": "architecture",
        "severity": "info",
        "description": "Actuator and motor control",
        "weight": 1.2,
        "confidence": 0.86
    },
    
    # Security
    "device_authentication": {
        "regex": r"(X\.509|certificate|device.*cert|mutual.*TLS|mTLS|device.*key)",
        "category": "security",
        "severity": "high",
        "description": "Device authentication using certificates",
        "weight": 1.6,
        "confidence": 0.88
    },
    "secure_boot": {
        "regex": r"(secure.*boot|verified.*boot|signature.*verification|trusted.*execution)",
        "category": "security",
        "severity": "high",
        "description": "Secure boot implementation",
        "weight": 1.7,
        "confidence": 0.82
    },
    
    # Edge Computing
    "edge_processing": {
        "regex": r"(edge.*compute|local.*processing|fog.*computing|edge.*inference|gateway)",
        "category": "architecture",
        "severity": "info",
        "description": "Edge computing and local processing",
        "weight": 1.4,
        "confidence": 0.83
    },
}

# =============================================================================
# AR/VR PATTERNS (Lighter Implementation)
# =============================================================================

AR_VR_PATTERNS: Dict[str, Dict[str, Any]] = {
    # Framework Detection
    "unity_engine": {
        "regex": r"(UnityEngine|MonoBehaviour|GameObject|Transform|Instantiate|Destroy\()",
        "category": "framework",
        "severity": "info",
        "description": "Unity game engine usage",
        "weight": 1.3,
        "confidence": 0.94
    },
    "unreal_engine": {
        "regex": r"(Unreal|UObject|AActor|UActorComponent|UPROPERTY|UFUNCTION)",
        "category": "framework",
        "severity": "info",
        "description": "Unreal Engine usage",
        "weight": 1.3,
        "confidence": 0.93
    },
    "webxr": {
        "regex": r"(WebXR|XRSession|XRFrame|navigator\.xr|immersive-vr|immersive-ar)",
        "category": "framework",
        "severity": "info",
        "description": "WebXR API for web-based XR",
        "weight": 1.4,
        "confidence": 0.91
    },
    
    # 3D Rendering
    "3d_rendering": {
        "regex": r"(Mesh|Shader|Material|Texture|Render|Camera|Light|Scene)",
        "category": "architecture",
        "severity": "info",
        "description": "3D rendering components",
        "weight": 1.2,
        "confidence": 0.85
    },
    "spatial_mapping": {
        "regex": r"(spatial.*map|mesh.*generation|environment.*scan|plane.*detection|anchor)",
        "category": "architecture",
        "severity": "info",
        "description": "Spatial mapping and environment understanding",
        "weight": 1.5,
        "confidence": 0.84
    },
    
    # Input & Tracking
    "hand_tracking": {
        "regex": r"(hand.*tracking|gesture|finger.*position|palm|pinch|grab|hand.*model)",
        "category": "architecture",
        "severity": "info",
        "description": "Hand tracking and gesture recognition",
        "weight": 1.4,
        "confidence": 0.86
    },
    "head_tracking": {
        "regex": r"(head.*tracking|pose.*estimation|6DOF|3DOF|orientation|position.*tracking)",
        "category": "architecture",
        "severity": "info",
        "description": "Head and pose tracking",
        "weight": 1.3,
        "confidence": 0.87
    },
    "eye_tracking": {
        "regex": r"(eye.*tracking|gaze|foveated|pupil|fixation|saccade)",
        "category": "architecture",
        "severity": "info",
        "description": "Eye tracking functionality",
        "weight": 1.4,
        "confidence": 0.85
    },
    
    # AR Specific
    "image_tracking": {
        "regex": r"(image.*target|marker.*tracking|ARImageAnchor|TrackedImage|reference.*image)",
        "category": "architecture",
        "severity": "info",
        "description": "Image target tracking for AR",
        "weight": 1.3,
        "confidence": 0.88
    },
    "object_occlusion": {
        "regex": r"(occlusion|depth.*buffer|z-buffer|environmental.*occlusion|people.*occlusion)",
        "category": "architecture",
        "severity": "info",
        "description": "AR occlusion handling",
        "weight": 1.3,
        "confidence": 0.84
    },
    
    # Performance
    "frame_optimization": {
        "regex": r"(frame.*rate|FPS|vsync|latency|motion.*sickness|reprojection|foveated.*render)",
        "category": "best_practice",
        "severity": "medium",
        "description": "Frame rate and latency optimization",
        "weight": 1.5,
        "confidence": 0.82
    },
    "lod_system": {
        "regex": r"(LOD|level.*of.*detail|distance.*culling|frustum.*cull|occlusion.*cull)",
        "category": "best_practice",
        "severity": "info",
        "description": "Level of Detail optimization system",
        "weight": 1.3,
        "confidence": 0.85
    },
}

# =============================================================================
# PATTERN AGGREGATION
# =============================================================================

ALL_DOMAIN_PATTERNS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "web3": WEB3_PATTERNS,
    "ml_ai": ML_AI_PATTERNS,
    "fintech": FINTECH_PATTERNS,
    "iot": IOT_PATTERNS,
    "ar_vr": AR_VR_PATTERNS,
}

# File extensions by domain
DOMAIN_FILE_EXTENSIONS: Dict[str, list] = {
    "web3": [".sol", ".vy", ".rs", ".move", ".ts", ".js", ".json"],
    "ml_ai": [".py", ".ipynb", ".yaml", ".yml", ".json", ".onnx"],
    "fintech": [".py", ".java", ".ts", ".js", ".go", ".cs", ".rb"],
    "iot": [".py", ".c", ".cpp", ".h", ".ino", ".rs", ".go"],
    "ar_vr": [".cs", ".cpp", ".js", ".ts", ".shader", ".hlsl", ".glsl"],
}

# Domain keywords for auto-detection
DOMAIN_KEYWORDS: Dict[str, list] = {
    "web3": [
        "blockchain", "ethereum", "solidity", "smart contract", "defi",
        "nft", "token", "wallet", "web3", "crypto", "decentralized"
    ],
    "ml_ai": [
        "machine learning", "deep learning", "neural network", "tensorflow",
        "pytorch", "model", "training", "inference", "ai", "nlp", "computer vision"
    ],
    "fintech": [
        "payment", "banking", "finance", "transaction", "kyc", "aml",
        "compliance", "stripe", "fintech", "ledger", "accounting"
    ],
    "iot": [
        "iot", "sensor", "device", "mqtt", "embedded", "raspberry pi",
        "arduino", "gateway", "telemetry", "edge"
    ],
    "ar_vr": [
        "augmented reality", "virtual reality", "ar", "vr", "xr", "unity",
        "unreal", "3d", "immersive", "headset", "spatial"
    ],
}
