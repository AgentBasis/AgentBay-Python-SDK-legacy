#!/usr/bin/env python3
"""
Sliding TTL Demo

This script demonstrates the sliding TTL functionality where sessions
are kept alive as long as they are being accessed within the TTL window.
"""

import time
import logging
import os
from datetime import datetime, timedelta
from tracker import AgentPerformanceTracker, ConversationQuality

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def sliding_ttl_demo():
    """Demonstrate sliding TTL where sessions reset timer on access"""
    logger.info("=== Sliding TTL Demo ===")
    
    # Initialize with very short TTL for demo (0.02 hours = ~1.2 minutes)
    perf_tracker = AgentPerformanceTracker(
        base_url="https://your-backend-api.com",
        api_key=os.getenv('AGENT_TRACKER_API_KEY'),
        session_ttl_hours=0.02,  # 1.2 minutes for demo
        cleanup_interval_minutes=1,  # Check every minute
        logger=logger
    )
    
    try:
        logger.info("Creating test sessions...")
        
        # Create multiple sessions
        active_session = perf_tracker.start_conversation(
            agent_id="active_agent",
            user_id="user_active",
            metadata={"type": "will_be_accessed"}
        )
        
        inactive_session = perf_tracker.start_conversation(
            agent_id="inactive_agent", 
            user_id="user_inactive",
            metadata={"type": "will_not_be_accessed"}
        )
        
        logger.info(f"Active session: {active_session}")
        logger.info(f"Inactive session: {inactive_session}")
        
        # Check initial stats
        stats = perf_tracker.get_session_stats()
        logger.info(f"Initial stats: {stats}")
        
        # Simulate ongoing conversation by touching the active session
        logger.info("\n--- Simulating active conversation ---")
        for i in range(6):  # Run for 6 intervals
            time.sleep(20)  # Wait 20 seconds
            
            # Touch the active session (sliding TTL)
            if active_session:
                touched = perf_tracker.touch_session(active_session)
                logger.info(f"Minute {i+1}: Active session touched: {touched}")
                
                # Check if sessions are still active
                active_alive = perf_tracker.is_session_active(active_session)
                inactive_alive = perf_tracker.is_session_active(inactive_session)
                
                logger.info(f"  Active session alive: {active_alive}")
                logger.info(f"  Inactive session alive: {inactive_alive}")
                
                # Get TTL remaining for active session
                ttl_remaining = perf_tracker.get_session_ttl_remaining(active_session)
                logger.info(f"  Active session TTL remaining: {ttl_remaining:.3f} hours")
            
            # Check session stats
            stats = perf_tracker.get_session_stats()
            logger.info(f"  Session stats: {stats}")
            
            if stats['active_sessions'] == 0:
                logger.info("  All sessions expired!")
                break
        
        # Demonstrate different access patterns
        logger.info("\n--- Testing access patterns ---")
        
        # Create new sessions with different access patterns
        frequent_session = perf_tracker.start_conversation(
            agent_id="frequent_agent",
            user_id="user_frequent"
        )
        
        occasional_session = perf_tracker.start_conversation(
            agent_id="occasional_agent", 
            user_id="user_occasional"
        )
        
        if frequent_session and occasional_session:
            logger.info(f"Frequent access session: {frequent_session}")
            logger.info(f"Occasional access session: {occasional_session}")
            
            # Access frequent session every 30 seconds, occasional every 60 seconds
            for i in range(4):
                time.sleep(30)
                
                # Always touch frequent session
                perf_tracker.touch_session(frequent_session)
                logger.info(f"Interval {i+1}: Frequent session touched")
                
                # Touch occasional session every other interval
                if i % 2 == 0:
                    perf_tracker.touch_session(occasional_session)
                    logger.info(f"Interval {i+1}: Occasional session touched")
                else:
                    logger.info(f"Interval {i+1}: Occasional session NOT touched")
                
                # Check which sessions are still alive
                freq_alive = perf_tracker.is_session_active(frequent_session)
                occ_alive = perf_tracker.is_session_active(occasional_session)
                
                logger.info(f"  Frequent session alive: {freq_alive}")
                logger.info(f"  Occasional session alive: {occ_alive}")
                
                # Show detailed session stats
                stats = perf_tracker.get_session_stats()
                logger.info(f"  Stats: {stats['active_sessions']} active, "
                           f"avg idle: {stats['avg_idle_time_hours']}h, "
                           f"longest idle: {stats['longest_idle_time_hours']}h")
        
        # Test ending active sessions
        logger.info("\n--- Testing session ending ---")
        if frequent_session and perf_tracker.is_session_active(frequent_session):
            result = perf_tracker.end_conversation(
                session_id=frequent_session,
                quality_score=ConversationQuality.EXCELLENT,
                message_count=25
            )
            logger.info(f"Ended frequent session: {result}")
        
        # Final session stats
        final_stats = perf_tracker.get_session_stats()
        logger.info(f"Final session stats: {final_stats}")
        
    except Exception as e:
        logger.error(f"Error during sliding TTL demo: {e}")
    finally:
        perf_tracker.close()
        logger.info("Sliding TTL demo completed")

def ttl_vs_sliding_comparison():
    """Compare regular TTL vs sliding TTL behavior"""
    logger.info("\n=== TTL vs Sliding TTL Comparison ===")
    
    # This demo shows conceptual difference
    logger.info("Conceptual comparison:")
    logger.info("Regular TTL: Session expires X hours after creation, regardless of usage")
    logger.info("Sliding TTL: Session expires X hours after LAST ACCESS")
    logger.info("")
    
    perf_tracker = AgentPerformanceTracker(
        base_url="https://your-backend-api.com",
        session_ttl_hours=0.01,  # ~36 seconds
        cleanup_interval_minutes=1
    )
    
    try:
        # Create a session
        session_id = perf_tracker.start_conversation(
            agent_id="comparison_agent",
            user_id="comparison_user"
        )
        
        if session_id:
            logger.info(f"Created session: {session_id}")
            start_time = datetime.now()
            
            # Access the session every 20 seconds for 2 minutes
            for i in range(6):
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"Time elapsed: {elapsed:.1f}s")
                
                # Check if session is still active
                is_active = perf_tracker.is_session_active(session_id)
                ttl_remaining = perf_tracker.get_session_ttl_remaining(session_id)
                
                logger.info(f"  Session active: {is_active}")
                if ttl_remaining is not None:
                    logger.info(f"  TTL remaining: {ttl_remaining * 3600:.1f} seconds")
                
                if not is_active:
                    logger.info("  Session expired!")
                    break
                
                time.sleep(20)
            
            logger.info("With sliding TTL, this session stayed alive because we kept accessing it!")
            logger.info("With regular TTL, it would have expired after ~36 seconds regardless.")
        
    except Exception as e:
        logger.error(f"Error during comparison: {e}")
    finally:
        perf_tracker.close()

if __name__ == "__main__":
    if not os.getenv('AGENT_TRACKER_API_KEY'):
        logger.warning("No API key set, but demo will still show local functionality")
    
    sliding_ttl_demo()
    ttl_vs_sliding_comparison() 