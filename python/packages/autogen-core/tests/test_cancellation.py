import asyncio
import threading
from dataclasses import dataclass
from typing import Callable

import pytest
from autogen_core import (
    AgentId,
    AgentInstantiationContext,
    CancellationToken,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
)


@dataclass
class MessageType: ...


# Note for future reader:
# To do cancellation, only the token should be interacted with as a user
# If you cancel a future, it may not work as you expect.


class LongRunningAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("A long running agent")
        self.called = False
        self.cancelled = False

    @message_handler
    async def on_new_message(self, message: MessageType, ctx: MessageContext) -> MessageType:
        self.called = True
        sleep = asyncio.ensure_future(asyncio.sleep(100))
        ctx.cancellation_token.link_future(sleep)
        try:
            await sleep
            return MessageType()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class NestingLongRunningAgent(RoutedAgent):
    def __init__(self, nested_agent: AgentId) -> None:
        super().__init__("A nesting long running agent")
        self.called = False
        self.cancelled = False
        self._nested_agent = nested_agent

    @message_handler
    async def on_new_message(self, message: MessageType, ctx: MessageContext) -> MessageType:
        self.called = True
        response = self.send_message(message, self._nested_agent, cancellation_token=ctx.cancellation_token)
        try:
            val = await response
            assert isinstance(val, MessageType)
            return val
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.mark.asyncio
async def test_cancellation_with_token() -> None:
    runtime = SingleThreadedAgentRuntime()

    await LongRunningAgent.register(runtime, "long_running", LongRunningAgent)
    agent_id = AgentId("long_running", key="default")
    token = CancellationToken()
    response = asyncio.create_task(runtime.send_message(MessageType(), recipient=agent_id, cancellation_token=token))
    assert not response.done()

    while runtime.unprocessed_messages_count == 0:
        await asyncio.sleep(0.01)

    await runtime._process_next()  # type: ignore

    token.cancel()

    with pytest.raises(asyncio.CancelledError):
        await response

    assert response.done()
    long_running_agent = await runtime.try_get_underlying_agent_instance(agent_id, type=LongRunningAgent)
    assert long_running_agent.called
    assert long_running_agent.cancelled


@pytest.mark.asyncio
async def test_nested_cancellation_only_outer_called() -> None:
    runtime = SingleThreadedAgentRuntime()

    await LongRunningAgent.register(runtime, "long_running", LongRunningAgent)
    await NestingLongRunningAgent.register(
        runtime,
        "nested",
        lambda: NestingLongRunningAgent(AgentId("long_running", key=AgentInstantiationContext.current_agent_id().key)),
    )

    long_running_id = AgentId("long_running", key="default")
    nested_id = AgentId("nested", key="default")
    token = CancellationToken()
    response = asyncio.create_task(runtime.send_message(MessageType(), nested_id, cancellation_token=token))
    assert not response.done()

    while runtime.unprocessed_messages_count == 0:
        await asyncio.sleep(0.01)

    await runtime._process_next()  # type: ignore
    token.cancel()

    with pytest.raises(asyncio.CancelledError):
        await response

    assert response.done()
    nested_agent = await runtime.try_get_underlying_agent_instance(nested_id, type=NestingLongRunningAgent)
    assert nested_agent.called
    assert nested_agent.cancelled
    long_running_agent = await runtime.try_get_underlying_agent_instance(long_running_id, type=LongRunningAgent)
    assert long_running_agent.called is False
    assert long_running_agent.cancelled is False


@pytest.mark.asyncio
async def test_nested_cancellation_inner_called() -> None:
    runtime = SingleThreadedAgentRuntime()

    await LongRunningAgent.register(runtime, "long_running", LongRunningAgent)
    await NestingLongRunningAgent.register(
        runtime,
        "nested",
        lambda: NestingLongRunningAgent(AgentId("long_running", key=AgentInstantiationContext.current_agent_id().key)),
    )

    long_running_id = AgentId("long_running", key="default")
    nested_id = AgentId("nested", key="default")

    token = CancellationToken()
    response = asyncio.create_task(runtime.send_message(MessageType(), nested_id, cancellation_token=token))
    assert not response.done()

    while runtime.unprocessed_messages_count == 0:
        await asyncio.sleep(0.01)

    await runtime._process_next()  # type: ignore
    # allow the inner agent to process
    await runtime._process_next()  # type: ignore
    token.cancel()

    with pytest.raises(asyncio.CancelledError):
        await response

    assert response.done()
    nested_agent = await runtime.try_get_underlying_agent_instance(nested_id, type=NestingLongRunningAgent)
    assert nested_agent.called
    assert nested_agent.cancelled
    long_running_agent = await runtime.try_get_underlying_agent_instance(long_running_id, type=LongRunningAgent)
    assert long_running_agent.called
    assert long_running_agent.cancelled


async def _cancel_on_thread(action: Callable[[], object]) -> list[Exception]:
    """Run a token operation on another thread and collect exceptions it raises."""
    errors: list[Exception] = []

    def _run() -> None:
        try:
            action()
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=_run)
    worker.start()
    await asyncio.to_thread(worker.join, 2)
    assert worker.is_alive() is False
    return errors


@pytest.mark.asyncio
async def test_link_future_cancel_on_owning_loop_is_immediate() -> None:
    """Cancellation on the future's loop still marks the future cancelled immediately."""
    loop = asyncio.get_running_loop()
    token = CancellationToken()
    future: asyncio.Future[None] = loop.create_future()
    assert token.link_future(future) is future
    token.cancel()
    assert future.cancelled()

    already_cancelled = CancellationToken()
    already_cancelled.cancel()
    linked_late: asyncio.Future[None] = loop.create_future()
    already_cancelled.link_future(linked_late)
    assert linked_late.cancelled()


@pytest.mark.asyncio
async def test_cancel_linked_future_from_other_thread() -> None:
    """Cancelling from another thread must cancel the linked future on its own loop."""
    loop = asyncio.get_running_loop()
    previous_debug = loop.get_debug()
    loop.set_debug(True)
    try:
        token = CancellationToken()
        future: asyncio.Future[None] = loop.create_future()
        callback_called = asyncio.Event()

        def _on_done(_future: asyncio.Future[None]) -> None:
            callback_called.set()

        future.add_done_callback(_on_done)
        assert token.link_future(future) is future

        errors = await _cancel_on_thread(token.cancel)

        assert errors == []
        await asyncio.wait_for(callback_called.wait(), timeout=2)
        assert future.cancelled()
    finally:
        loop.set_debug(previous_debug)


@pytest.mark.asyncio
async def test_link_future_to_already_cancelled_token_from_other_thread() -> None:
    """Linking a future from another thread must still cancel it when the token is already cancelled."""
    loop = asyncio.get_running_loop()
    previous_debug = loop.get_debug()
    loop.set_debug(True)
    try:
        token = CancellationToken()
        token.cancel()
        future: asyncio.Future[None] = loop.create_future()
        callback_called = asyncio.Event()

        def _on_done(_future: asyncio.Future[None]) -> None:
            callback_called.set()

        future.add_done_callback(_on_done)

        errors = await _cancel_on_thread(lambda: token.link_future(future))

        assert errors == []
        await asyncio.wait_for(callback_called.wait(), timeout=2)
        assert future.cancelled()
    finally:
        loop.set_debug(previous_debug)


@pytest.mark.asyncio
async def test_cancel_linked_future_on_closed_loop_does_not_raise() -> None:
    """Cancelling a future whose loop is already closed must not raise."""
    loop = asyncio.new_event_loop()
    try:
        future: asyncio.Future[None] = loop.create_future()

        def _on_done(_future: asyncio.Future[None]) -> None:
            return None

        future.add_done_callback(_on_done)
        loop.close()

        token = CancellationToken()
        token.link_future(future)
        token.cancel()
        assert future.cancelled()
    finally:
        if not loop.is_closed():
            loop.close()
