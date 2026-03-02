"""
AURORA-X Message Broker.

Abstract message broker with pluggable backends:
- InMemoryBroker: asyncio.Queue-based for development
- KafkaBroker: aiokafka-based for production
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger("aurora_x.ingestion.broker")


class MessageBroker(ABC):
    """Abstract message broker interface."""

    @abstractmethod
    async def start(self):
        ...

    @abstractmethod
    async def stop(self):
        ...

    @abstractmethod
    async def publish(self, topic: str, key: str, value: Any):
        ...

    @abstractmethod
    async def consume_batch(
        self, topic: str, max_batch: int = 50, timeout_ms: int = 100
    ) -> List[Dict[str, Any]]:
        ...


class InMemoryBroker(MessageBroker):
    """In-memory message broker using asyncio queues.

    Each topic is an asyncio.Queue. Suitable for development and testing.
    """

    def __init__(self):
        self._topics: Dict[str, asyncio.Queue] = defaultdict(
            lambda: asyncio.Queue(maxsize=10000)
        )
        self._running = False

    async def start(self):
        self._running = True
        logger.info("InMemoryBroker started")

    async def stop(self):
        self._running = False
        logger.info("InMemoryBroker stopped")

    async def publish(self, topic: str, key: str, value: Any):
        if not self._running:
            return
        try:
            self._topics[topic].put_nowait(value)
        except asyncio.QueueFull:
            # Drop oldest event (backpressure)
            try:
                self._topics[topic].get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._topics[topic].put_nowait(value)

    async def consume_batch(
        self, topic: str, max_batch: int = 50, timeout_ms: int = 100
    ) -> List[Dict[str, Any]]:
        queue = self._topics[topic]
        batch = []
        deadline = asyncio.get_event_loop().time() + timeout_ms / 1000

        while len(batch) < max_batch:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=remaining)
                batch.append(item)
            except asyncio.TimeoutError:
                break
        return batch


class KafkaBroker(MessageBroker):
    """Kafka broker using aiokafka. For production deployment."""

    def __init__(self, bootstrap_servers: str, topic_prefix: str = "aurora"):
        self.bootstrap_servers = bootstrap_servers
        self.topic_prefix = topic_prefix
        self._producer = None
        self._consumer = None

    async def start(self):
        try:
            from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
            )
            await self._producer.start()
            logger.info("KafkaBroker producer started at %s", self.bootstrap_servers)
        except ImportError:
            logger.error("aiokafka not installed. Use InMemoryBroker for development.")
            raise

    async def stop(self):
        if self._producer:
            await self._producer.stop()
        if self._consumer:
            await self._consumer.stop()
        logger.info("KafkaBroker stopped")

    async def publish(self, topic: str, key: str, value: Any):
        full_topic = f"{self.topic_prefix}.{topic}"
        await self._producer.send(full_topic, value=value, key=key)

    async def consume_batch(
        self, topic: str, max_batch: int = 50, timeout_ms: int = 100
    ) -> List[Dict[str, Any]]:
        if self._consumer is None:
            from aiokafka import AIOKafkaConsumer

            full_topic = f"{self.topic_prefix}.{topic}"
            self._consumer = AIOKafkaConsumer(
                full_topic,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="latest",
                group_id="aurora-x-processor",
            )
            await self._consumer.start()

        batch = []
        data = await self._consumer.getmany(
            timeout_ms=timeout_ms, max_records=max_batch
        )
        for tp, messages in data.items():
            for msg in messages:
                batch.append(msg.value)
        return batch


def create_broker(config: Dict[str, Any]) -> MessageBroker:
    """Factory function to create the appropriate broker based on config."""
    backend = config.get("broker", {}).get("backend", "memory")

    if backend == "kafka":
        bootstrap = config.get("broker", {}).get(
            "kafka_bootstrap", "localhost:9092"
        )
        prefix = config.get("broker", {}).get("topic_prefix", "aurora")
        logger.info("Creating KafkaBroker (servers=%s)", bootstrap)
        return KafkaBroker(bootstrap, prefix)
    else:
        logger.info("Creating InMemoryBroker")
        return InMemoryBroker()
