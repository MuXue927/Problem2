import cProfile
import pytest

if __name__ == "__main__":
    # Run the single integration test under cProfile and write profile to logs-alns/profile_after_patch.prof
    cProfile.run('pytest.main(["-q", "ALNSCode/test/test_adaptive_degree_integration.py::test_adaptive_degree_integration"])', 'logs-alns/profile_after_patch.prof')
    print("Profiling run complete, profile saved to logs-alns/profile_after_patch.prof")
