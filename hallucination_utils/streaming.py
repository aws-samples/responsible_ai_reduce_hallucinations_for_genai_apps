# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utilities to simplify working with streaming data and events

Strands-Agents is streaming-first, but we use the tools in this module to help hide some of that
complexity in the workshop exercises.
"""

# Python Built-Ins:
from asyncio import FIRST_COMPLETED, Task, create_task, wait
from logging import getLogger
from typing import AsyncIterator, AsyncIterable, TypeVar


logger = getLogger(__name__)

_T = TypeVar("_T")


async def list_to_async_iterable(items: list[_T]) -> AsyncIterable[_T]:
    for item in items:
        yield item


async def collect_async_iterables(*iterables: AsyncIterable[_T]) -> list[list[_T]]:
    """Concurrently consume multiple iterables and collect their results into lists

    Like `collect_async_iterators`, but for things that *can* be iterated (multiple times) rather
    than things that *are being* iterated (single-shot).
    """
    return await collect_async_iterators(
        *[iterable.__aiter__() for iterable in iterables]
    )


async def collect_async_iterators(*iterators: AsyncIterator[_T]) -> list[list[_T]]:
    """Concurrently consume multiple iterators and collect their results into lists

    We use this to consolidate multiple streaming LLM call results for non-streaming analysis.

    Inspired by https://stackoverflow.com/a/76643550

    Parameters
    ----------
    *iterators :
        Event/data sources to iterate over

    Returns
    -------
    results :
        Nested list of yielded values by 1/ source iterator and 2/ sequence of result.
    """
    results = [[] for _ in iterators]

    async def await_next(iterator: AsyncIterator[_T]) -> _T:
        """Turn an awaitable into a coroutine for `asyncio.create_task`."""
        return await iterator.__anext__()

    def as_task(iterator: AsyncIterator[_T]) -> Task[_T]:
        """Turn an iterator into an asyncio task"""
        return create_task(await_next(iterator))

    # Create a task for each iterator, keyed on the iterator.
    next_tasks = {iterator: as_task(iterator) for iterator in iterators}
    # As iterators are exhausted, they'll be removed from the mapping.
    # Repeat for as long as any are NOT exhausted.
    while next_tasks:
        # Wait until one of the iterators yields or errors out:
        # (This also returns pending tasks, but we've got those in our mapping)
        done, _ = await wait(next_tasks.values(), return_when=FIRST_COMPLETED)

        for task in done:
            # Identify the iterator.
            iterator = next(it for it, t in next_tasks.items() if t == task)
            ix = iterators.index(iterator)
            try:
                result = task.result()
                logger.debug(
                    "(Iterator %s of %s) yielded %s", ix, len(iterators), result
                )
                results[ix].append(result)
            except StopAsyncIteration:
                # This iterator has finished - clear it from the map
                del next_tasks[iterator]
            # In this case we don't catch any other errors - so any error will be propagated up
            else:
                # Iterator still going - queue the next inspection:
                next_tasks[iterator] = as_task(iterator)

    # At this point, all iterators are exhausted.
    return results
