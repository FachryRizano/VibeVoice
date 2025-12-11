import json
import asyncio
import aio_pika
import logging
from datetime import datetime
from aio_pika import Message
QUEUE_ARGUMENTS = {
    "x-dead-letter-exchange" : "dlx_exchange",
    "x-dead-letter-routing-key" : "dlx_tts_gateway_key",
    "x-message-ttl" : 60000,
    "x-queue-type": "quorum"
}
# Set logging
logger = logging.getLogger(__name__)

class RabbitMQHandler:
    """RabbitMQ client handler for receiving and sending messages."""
    def __init__(self, endpoint: str, user: str, password: str, vhost: str, port: int = 5672, arguments:dict = QUEUE_ARGUMENTS, prefech_count: int = 16):
        self.endpoint = endpoint
        self.user = user
        self.password = password
        self.vhost = vhost
        self.port = port
        self.connection = None  # Store connection for reuse
        self.arguments = arguments
        self.prefetch_count = prefech_count

    async def connect(self):
        """Connect to RabbitMQ endpoint."""
        if self.connection is None or self.connection.is_closed:
            self.connection = await aio_pika.connect_robust(
                host=self.endpoint,
                port=self.port,
                login=self.user,
                password=self.password,
                virtualhost=self.vhost
            )
            logger.info("Connected to RabbitMQ")
        return self.connection

    async def start_listener(self, queue_name: str, callback_function):
        """Start listening to a specific queue."""
        connection = await self.connect()

        async with connection:
            logger.info("Starting RabbitMQ listener...")
            channel = await connection.channel()

            # Set QoS to limit the number of unacknowledged messages
            await channel.set_qos(prefetch_count=self.prefetch_count)

            # Declare the queue
            queue = await channel.declare_queue(
                queue_name,
                durable=True,
                arguments=self.arguments
            )

            logger.info(f"Listening to queue: {queue_name}")
            await queue.consume(callback_function)

            try:
                # Keep the listener running
                await asyncio.Future()
            except asyncio.CancelledError:
                logger.info("Listener task cancelled")
            finally:
                await connection.close()
                logger.info("RabbitMQ connection closed")

    async def publish_message(self, payload: dict, routing_key: str, exchange_name: str, exchange_type: str):
        """Publish a message to RabbitMQ."""
        if not isinstance(payload, str):
            payload = json.dumps(payload)
        encoded_payload = payload.encode()

        connection = await self.connect()

        async with connection:
            channel = await connection.channel()

            # Declare the exchange
            exchange = await channel.declare_exchange(
                name=exchange_name,
                type=exchange_type,
                durable=True
            )

            # Wrap the message in the Message class
            message_payload = Message(
                content_type="application/json",
                body=encoded_payload
            )

            # Publish the message
            await exchange.publish(
                message=message_payload,
                routing_key=routing_key
            )
            logger.info(f"Message published to exchange '{exchange_name}' with routing key '{routing_key}'")

    async def publish_message_dummy(self, exchange_name: str, exchange_type: str, routing_key: str, sentence: str, payload: bool):
        """Publish dummy data to RabbitMQ."""
        connection = await self.connect()

        async with connection:
            channel = await connection.channel()

            # Declare the exchange
            exchange = await channel.declare_exchange(
                name=exchange_name,
                type=exchange_type,
                durable=True
            )

            # Prepare the payload
            if payload:
                payload = [{"blob_name": "", "url": ""}]
            else:
                payload = [{}]

            # Create the message
            message = {
                "user": {
                    "room_id": "262285944",
                    "is_bot": False,
                    "is_handled_by_bot": True,
                    "is_resolved": False,
                    "sender_email": "MALIK@malik.co",
                    "epoch_timestamp": str(datetime.now().timestamp()),
                    "name": "malik",
                },
                "message": {
                    "timestamp": str(datetime.now().timestamp()),
                    "message": sentence,
                    "payload": payload,
                    "content": {}
                }
            }

            # Encode the message
            encoded_message = Message(json.dumps(message).encode())

            # Publish the message
            await exchange.publish(
                message=encoded_message,
                routing_key=routing_key
            )
            logger.info(f"Dummy message published to exchange '{exchange_name}' with routing key '{routing_key}'")