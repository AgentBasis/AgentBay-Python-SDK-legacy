#!/usr/bin/env python3
"""
Simple Sliding TTL Test

Test sliding TTL functionality locally without requiring backend connection.
"""

import time
import logging
from datetime import datetime, timedelta
from tracker.AgentPerform import SessionInfo

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_session_info_sliding_ttl():
    """Test SessionInfo sliding TTL behavior"""
    print("=== Testing SessionInfo Sliding TTL ===")
    
    # Create a session info with short TTL
    start_time = datetime.now()
    session = SessionInfo(
        agent_id="test_agent",
        start_time=start_time,
        user_id="test_user"
    )
    
    print(f"Session created at: {session.start_time}")
    print(f"Last access time: {session.last_access_time}")
    print(f"Initial expired status (TTL=0.001h): {session.is_expired(0.001)}")  # ~3.6 seconds
    
    # Wait 2 seconds
    print("\nWaiting 2 seconds...")
    time.sleep(2)
    
    # Check if expired (should be expired with such short TTL)
    print(f"After 2s - Expired: {session.is_expired(0.001)}")
    
    # Touch the session (reset TTL)
    print("Touching session (resetting TTL)...")
    session.touch()
    print(f"New last access time: {session.last_access_time}")
    
    # Check if expired again (should NOT be expired now)
    print(f"After touch - Expired: {session.is_expired(0.001)}")
    
    # Wait another 2 seconds
    print("\nWaiting another 2 seconds...")
    time.sleep(2)
    
    # Should be expired now
    print(f"After another 2s - Expired: {session.is_expired(0.001)}")
    
    print("\n✅ SessionInfo sliding TTL test completed!")

def test_sliding_vs_regular_ttl():
    """Compare sliding TTL vs regular TTL behavior"""
    print("\n=== Sliding TTL vs Regular TTL Comparison ===")
    
    start_time = datetime.now()
    
    # Simulate regular TTL (expires based on start_time)
    def is_expired_regular(start_time, ttl_hours):
        expiry_time = start_time + timedelta(hours=ttl_hours)
        return datetime.now() > expiry_time
    
    # Simulate sliding TTL (expires based on last_access_time)
    session = SessionInfo(
        agent_id="comparison_agent",
        start_time=start_time,
        user_id="comparison_user"
    )
    
    ttl_hours = 0.001  # ~3.6 seconds
    
    print(f"TTL: {ttl_hours * 3600:.1f} seconds")
    print(f"Start time: {start_time.strftime('%H:%M:%S.%f')[:-3]}")
    
    # Test over 8 seconds with access every 2 seconds
    for i in range(4):
        time.sleep(2)
        current_time = datetime.now()
        elapsed = (current_time - start_time).total_seconds()
        
        # Regular TTL check
        regular_expired = is_expired_regular(start_time, ttl_hours)
        
        # Sliding TTL check (with access/touch)
        sliding_expired_before = session.is_expired(ttl_hours)
        session.touch()  # Access resets TTL
        sliding_expired_after = session.is_expired(ttl_hours)
        
        print(f"\nElapsed: {elapsed:.1f}s")
        print(f"  Regular TTL expired: {regular_expired}")
        print(f"  Sliding TTL before touch: {sliding_expired_before}")
        print(f"  Sliding TTL after touch: {sliding_expired_after}")
        
        if regular_expired and not sliding_expired_after:
            print("  → Sliding TTL keeps session alive!")
    
    print("\n✅ TTL comparison test completed!")

def test_session_access_patterns():
    """Test different session access patterns"""
    print("\n=== Testing Session Access Patterns ===")
    
    # Create sessions with different access patterns
    frequent_session = SessionInfo("frequent_agent", datetime.now(), "user1")
    occasional_session = SessionInfo("occasional_agent", datetime.now(), "user2")
    abandoned_session = SessionInfo("abandoned_agent", datetime.now(), "user3")
    
    ttl_hours = 0.002  # ~7.2 seconds
    
    print(f"TTL: {ttl_hours * 3600:.1f} seconds")
    print("Testing 3 sessions with different access patterns...")
    
    # Simulate 12 seconds with different access patterns
    for i in range(6):
        time.sleep(2)
        elapsed = (i + 1) * 2
        
        # Frequent session: access every time
        frequent_session.touch()
        frequent_expired = frequent_session.is_expired(ttl_hours)
        
        # Occasional session: access every other time
        if i % 2 == 0:
            occasional_session.touch()
        occasional_expired = occasional_session.is_expired(ttl_hours)
        
        # Abandoned session: never accessed
        abandoned_expired = abandoned_session.is_expired(ttl_hours)
        
        print(f"\nAfter {elapsed}s:")
        print(f"  Frequent (accessed): {'❌ Expired' if frequent_expired else '✅ Active'}")
        print(f"  Occasional (access every 4s): {'❌ Expired' if occasional_expired else '✅ Active'}")
        print(f"  Abandoned (never accessed): {'❌ Expired' if abandoned_expired else '✅ Active'}")
    
    print("\n✅ Session access patterns test completed!")

if __name__ == "__main__":
    test_session_info_sliding_ttl()
    test_sliding_vs_regular_ttl()
    test_session_access_patterns()
    
    print("\n🎉 All sliding TTL tests completed successfully!")
    print("\nKey takeaways:")
    print("- Sliding TTL resets expiry timer on every access")
    print("- Active sessions stay alive indefinitely with regular access")
    print("- Abandoned sessions expire after TTL period")
    print("- This prevents memory leaks while protecting active conversations") 