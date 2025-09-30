"""
Background metrics sender for async processing of metrics batches.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
from collections import deque

from .models import TranscriptMetrics, MonitoringBatch
from .vertex_client import VertexAIClient
from .config import MonitoringConfig
from .exceptions import BackgroundSenderError

logger = logging.getLogger(__name__)


class BackgroundMetricsSender:
    """Handles background sending of metrics to Vertex AI."""

    def __init__(self, config: MonitoringConfig, vertex_client: VertexAIClient):
        self.config = config
        self.vertex_client = vertex_client

        # Queue for pending metrics
        self._metrics_queue: deque = deque()
        self._batch_queue: deque = deque()

        # Background task management
        self._sender_task: Optional[asyncio.Task] = None
        self._batch_processor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Offline storage
        self._offline_storage_path = Path("/tmp/claude/transcript_metrics_offline")
        if config.offline_mode:
            self._offline_storage_path.mkdir(parents=True, exist_ok=True)

        # Stats
        self._stats = {
            "metrics_queued": 0,
            "batches_sent": 0,
            "send_failures": 0,
            "offline_saves": 0,
            "last_send_time": None,
            "queue_size": 0,
            "batch_queue_size": 0
        }

    async def start(self):
        """Start background processing tasks."""
        if self._running:
            return

        self._running = True
        logger.info("Starting background metrics sender")

        # Start batch processor
        self._batch_processor_task = asyncio.create_task(self._batch_processor())

        # Start sender task
        self._sender_task = asyncio.create_task(self._sender_loop())

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Load any offline metrics
        if self.config.offline_mode:
            await self._load_offline_metrics()

    async def stop(self):
        """Stop background processing and cleanup."""
        logger.info("Stopping background metrics sender")
        self._running = False

        # Cancel tasks
        tasks = [self._sender_task, self._batch_processor_task, self._cleanup_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Process remaining metrics if possible
        await self._process_remaining_metrics()

        logger.info("Background metrics sender stopped")

    async def queue_metrics(self, metrics: TranscriptMetrics):
        """Queue metrics for background sending."""
        self._metrics_queue.append(metrics)
        self._stats["metrics_queued"] += 1
        self._stats["queue_size"] = len(self._metrics_queue)

        logger.debug(f"Queued metrics for request {metrics.request_id}")

    async def _batch_processor(self):
        """Process metrics queue into batches."""
        while self._running:
            try:
                if len(self._metrics_queue) >= self.config.batch_size:
                    # Create batch from queued metrics
                    batch_metrics = []
                    for _ in range(min(self.config.batch_size, len(self._metrics_queue))):
                        if self._metrics_queue:
                            batch_metrics.append(self._metrics_queue.popleft())

                    if batch_metrics:
                        batch = MonitoringBatch(
                            batch_id=f"batch_{int(time.time())}_{len(batch_metrics)}",
                            metrics=batch_metrics
                        )
                        self._batch_queue.append(batch)
                        self._stats["queue_size"] = len(self._metrics_queue)
                        self._stats["batch_queue_size"] = len(self._batch_queue)

                        logger.debug(f"Created batch {batch.batch_id} with {len(batch_metrics)} metrics")

                # Also create batches based on timeout
                elif self._metrics_queue and self._should_create_timeout_batch():
                    batch_metrics = list(self._metrics_queue)
                    self._metrics_queue.clear()

                    batch = MonitoringBatch(
                        batch_id=f"timeout_batch_{int(time.time())}_{len(batch_metrics)}",
                        metrics=batch_metrics
                    )
                    self._batch_queue.append(batch)
                    self._stats["queue_size"] = 0
                    self._stats["batch_queue_size"] = len(self._batch_queue)

                    logger.debug(f"Created timeout batch {batch.batch_id} with {len(batch_metrics)} metrics")

                await asyncio.sleep(1)  # Check every second

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(5)

    def _should_create_timeout_batch(self) -> bool:
        """Check if we should create a batch based on timeout."""
        if not self._metrics_queue:
            return False

        # Get the oldest metric's timestamp
        oldest_metric = self._metrics_queue[0]
        time_diff = datetime.now() - oldest_metric.timestamp
        return time_diff.total_seconds() >= self.config.batch_timeout_seconds

    async def _sender_loop(self):
        """Main sender loop for processing batches."""
        while self._running:
            try:
                if self._batch_queue:
                    batch = self._batch_queue.popleft()
                    self._stats["batch_queue_size"] = len(self._batch_queue)

                    success = await self._send_batch_with_retry(batch)

                    if success:
                        self._stats["batches_sent"] += 1
                        self._stats["last_send_time"] = datetime.now()
                        logger.debug(f"Successfully sent batch {batch.batch_id}")
                    else:
                        self._stats["send_failures"] += 1
                        logger.warning(f"Failed to send batch {batch.batch_id}")

                        # Save to offline storage if enabled
                        if self.config.offline_mode:
                            await self._save_batch_offline(batch)

                else:
                    await asyncio.sleep(2)  # Wait if no batches

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sender loop: {e}")
                await asyncio.sleep(5)

    async def _send_batch_with_retry(self, batch: MonitoringBatch) -> bool:
        """Send batch with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                if not self.config.send_to_vertex:
                    # If vertex sending is disabled, just log
                    logger.info(f"Vertex AI disabled - would send batch {batch.batch_id}")
                    return True

                success = await self.vertex_client.send_metrics_batch(batch)
                if success:
                    return True

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for batch {batch.batch_id}: {e}")

                if attempt < self.config.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)

        return False

    async def _save_batch_offline(self, batch: MonitoringBatch):
        """Save batch to offline storage."""
        try:
            filename = f"batch_{batch.batch_id}_{int(time.time())}.json"
            filepath = self._offline_storage_path / filename

            batch_data = {
                "batch_id": batch.batch_id,
                "batch_timestamp": batch.batch_timestamp.isoformat(),
                "batch_size": batch.batch_size,
                "metrics": [metric.dict() for metric in batch.metrics]
            }

            with open(filepath, 'w') as f:
                json.dump(batch_data, f, indent=2, default=str)

            self._stats["offline_saves"] += 1
            logger.info(f"Saved batch {batch.batch_id} to offline storage: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save batch {batch.batch_id} offline: {e}")

    async def _load_offline_metrics(self):
        """Load and process offline stored metrics."""
        try:
            offline_files = list(self._offline_storage_path.glob("batch_*.json"))
            if not offline_files:
                return

            logger.info(f"Found {len(offline_files)} offline batch files to process")

            for filepath in offline_files:
                try:
                    with open(filepath, 'r') as f:
                        batch_data = json.load(f)

                    # Recreate metrics objects
                    metrics = []
                    for metric_data in batch_data["metrics"]:
                        metrics.append(TranscriptMetrics(**metric_data))

                    # Recreate batch
                    batch = MonitoringBatch(
                        batch_id=batch_data["batch_id"],
                        metrics=metrics,
                        batch_timestamp=datetime.fromisoformat(batch_data["batch_timestamp"])
                    )

                    # Add to batch queue
                    self._batch_queue.append(batch)
                    self._stats["batch_queue_size"] = len(self._batch_queue)

                    logger.info(f"Loaded offline batch {batch.batch_id} with {len(metrics)} metrics")

                    # Remove processed file
                    filepath.unlink()

                except Exception as e:
                    logger.error(f"Failed to load offline batch from {filepath}: {e}")

        except Exception as e:
            logger.error(f"Error loading offline metrics: {e}")

    async def _cleanup_loop(self):
        """Periodic cleanup of old data and stats."""
        while self._running:
            try:
                # Clean up old offline files (older than 24 hours)
                if self.config.offline_mode:
                    await self._cleanup_old_offline_files()

                # Update stats
                self._stats["queue_size"] = len(self._metrics_queue)
                self._stats["batch_queue_size"] = len(self._batch_queue)

                # Sleep for 5 minutes
                await asyncio.sleep(300)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)

    async def _cleanup_old_offline_files(self):
        """Clean up offline files older than 24 hours."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            offline_files = list(self._offline_storage_path.glob("batch_*.json"))

            for filepath in offline_files:
                if datetime.fromtimestamp(filepath.stat().st_mtime) < cutoff_time:
                    filepath.unlink()
                    logger.debug(f"Cleaned up old offline file: {filepath}")

        except Exception as e:
            logger.error(f"Error cleaning up offline files: {e}")

    async def _process_remaining_metrics(self):
        """Process any remaining metrics before shutdown."""
        try:
            # Process remaining individual metrics into a final batch
            if self._metrics_queue:
                final_metrics = list(self._metrics_queue)
                self._metrics_queue.clear()

                final_batch = MonitoringBatch(
                    batch_id=f"shutdown_batch_{int(time.time())}",
                    metrics=final_metrics
                )

                logger.info(f"Processing final batch with {len(final_metrics)} metrics")
                await self._send_batch_with_retry(final_batch)

            # Process remaining batches
            while self._batch_queue:
                batch = self._batch_queue.popleft()
                logger.info(f"Processing remaining batch {batch.batch_id}")
                await self._send_batch_with_retry(batch)

        except Exception as e:
            logger.error(f"Error processing remaining metrics: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current sender statistics."""
        stats = self._stats.copy()
        stats.update({
            "running": self._running,
            "queue_size": len(self._metrics_queue),
            "batch_queue_size": len(self._batch_queue)
        })
        return stats

    async def health_check(self) -> bool:
        """Perform health check on background sender."""
        try:
            return (
                self._running and
                (not self._sender_task or not self._sender_task.done()) and
                (not self._batch_processor_task or not self._batch_processor_task.done())
            )
        except Exception:
            return False

    def force_send_batch(self):
        """Force creation of a batch from current queue."""
        if self._metrics_queue:
            batch_metrics = list(self._metrics_queue)
            self._metrics_queue.clear()

            batch = MonitoringBatch(
                batch_id=f"forced_batch_{int(time.time())}",
                metrics=batch_metrics
            )
            self._batch_queue.append(batch)
            self._stats["queue_size"] = 0
            self._stats["batch_queue_size"] = len(self._batch_queue)

            logger.info(f"Force created batch {batch.batch_id} with {len(batch_metrics)} metrics")