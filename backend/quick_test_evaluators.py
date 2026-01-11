"""
Quick Integration Test for Domain Evaluators
=============================================

This script performs a quick end-to-end test of the domain evaluators.
Run directly to validate the implementation works correctly.

Usage:
    python quick_test_evaluators.py
"""

import asyncio
import tempfile
from pathlib import Path

# Import evaluator modules
from evaluators.models import DomainType
from evaluators.registry import get_registry
from evaluators.orchestrator import DomainOrchestrator


def create_test_repo() -> Path:
    """Create a temporary test repository with Web3 code."""
    tmpdir = tempfile.mkdtemp(prefix="evalx_test_")
    repo_path = Path(tmpdir)
    
    # Create contracts directory
    contracts_dir = repo_path / "contracts"
    contracts_dir.mkdir()
    
    # Create a Solidity file with Web3 patterns
    sol_file = contracts_dir / "Token.sol"
    sol_file.write_text("""
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title MyToken
 * @dev ERC20 Token with security features
 */
contract MyToken is ERC20, ReentrancyGuard, Ownable {
    
    mapping(address => uint256) private _stakes;
    
    event TokensStaked(address indexed user, uint256 amount);
    event TokensUnstaked(address indexed user, uint256 amount);
    
    constructor() ERC20("MyToken", "MTK") {}
    
    function stake(uint256 amount) external nonReentrant {
        require(amount > 0, "Cannot stake 0");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        
        _stakes[msg.sender] += amount;
        _transfer(msg.sender, address(this), amount);
        
        emit TokensStaked(msg.sender, amount);
    }
    
    function unstake(uint256 amount) external nonReentrant {
        require(_stakes[msg.sender] >= amount, "Insufficient stake");
        
        _stakes[msg.sender] -= amount;
        _transfer(address(this), msg.sender, amount);
        
        emit TokensUnstaked(msg.sender, amount);
    }
    
    function getStake(address account) external view returns (uint256) {
        return _stakes[account];
    }
}
    """)
    
    # Create README
    readme = repo_path / "README.md"
    readme.write_text("""
# MyToken - DeFi Token

A decentralized finance (DeFi) token built on Ethereum with staking capabilities.

## Features

- ERC20 compliant token
- Staking mechanism
- Reentrancy protection
- Owner access control

## Smart Contracts

The main contract is `Token.sol` which implements:
- Standard ERC20 functionality
- Stake/unstake mechanism
- Security features (ReentrancyGuard)

## Technology Stack

- Solidity ^0.8.0
- OpenZeppelin Contracts
- Hardhat/Truffle for development
    """)
    
    # Create package.json
    package = repo_path / "package.json"
    package.write_text("""
{
    "name": "mytoken",
    "version": "1.0.0",
    "dependencies": {
        "@openzeppelin/contracts": "^4.8.0",
        "ethers": "^5.7.0",
        "hardhat": "^2.12.0",
        "web3": "^1.8.0"
    }
}
    """)
    
    return repo_path


async def run_tests():
    """Run integration tests."""
    print("=" * 60)
    print("Domain Evaluators Quick Integration Test")
    print("=" * 60)
    
    # Test 1: Registry
    print("\n[Test 1] Registry Initialization...")
    registry = get_registry()
    assert len(registry) == 5, "Registry should have 5 domains"
    print(f"  ✓ Registry loaded with {len(registry)} domains")
    
    # Test 2: Get evaluators
    print("\n[Test 2] Evaluator Instantiation...")
    for domain in [DomainType.WEB3, DomainType.ML_AI, DomainType.FINTECH]:
        evaluator = registry.get_evaluator(domain)
        patterns = evaluator.get_patterns()
        print(f"  ✓ {domain.value}: {len(patterns)} patterns available")
    
    # Test 3: Orchestrator
    print("\n[Test 3] Orchestrator Initialization...")
    orchestrator = DomainOrchestrator()
    supported = orchestrator.get_supported_domains()
    assert len(supported) == 5, "Orchestrator should support 5 domains"
    print(f"  ✓ Orchestrator supports: {[d.value for d in supported]}")
    
    # Test 4: Create test repo and detect domain
    print("\n[Test 4] Domain Detection...")
    test_repo = create_test_repo()
    try:
        # Use detector_utils directly for testing
        from evaluators.detector_utils import detect_domain_from_files
        detected, confidence = detect_domain_from_files(str(test_repo))
        print(f"  ✓ Detected domain: {detected.value} with {confidence:.2%} confidence")
        assert detected == DomainType.WEB3, f"Should detect Web3 domain, got {detected.value}"
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(test_repo, ignore_errors=True)
    
    # Test 5: Pattern matching test
    print("\n[Test 5] Pattern Matching...")
    web3_evaluator = registry.get_evaluator(DomainType.WEB3)
    test_code = """
    contract Token is ERC20, ReentrancyGuard {
        modifier nonReentrant() {
            require(!locked);
            locked = true;
            _;
            locked = false;
        }
        
        function transfer(address to, uint256 amount) external nonReentrant {
            require(to != address(0), "Invalid address");
            emit Transfer(msg.sender, to, amount);
        }
    }
    """
    
    patterns = web3_evaluator.get_patterns()
    matches_found = 0
    for name, pattern_info in patterns.items():
        import re
        regex = pattern_info.get("regex", "")
        if re.search(regex, test_code, re.IGNORECASE | re.MULTILINE):
            matches_found += 1
    
    print(f"  ✓ Found {matches_found} pattern matches in test code")
    
    # Test 6: Evaluation info retrieval
    print("\n[Test 6] Evaluator Info...")
    info = registry.get_evaluator_info(DomainType.WEB3)
    assert "domain" in info
    assert "file_extensions" in info
    print(f"  ✓ Web3 evaluator info: {info['class_name']}")
    print(f"  ✓ Extensions: {info['file_extensions'][:5]}...")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    result = asyncio.run(run_tests())
    exit(0 if result else 1)
