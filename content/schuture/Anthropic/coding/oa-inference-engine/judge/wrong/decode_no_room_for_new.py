# expect: WA
# Decode allocates a new block only once the sequence's tokens no longer fit its blocks, i.e. after the token
# that needed the room was already generated, so the block arrives one step late.
from collections import deque, OrderedDict
from enum import Enum
from typing import Optional

VOCAB_SIZE = 17          # token ids are 0 .. VOCAB_SIZE - 1
EOS_TOKEN = 16            # generating this token ends a sequence
BLOCK_SIZE = 4            # tokens held by one KV-cache block


def next_token(token_ids: list[int]) -> int:
    """Stand-in for a language model's forward pass: the next token is a fixed hash of the whole
    token sequence so far (the prompt and everything generated after it). Deterministic, needs no
    weights. token_ids must be non-empty."""
    h = 0
    for t in token_ids:
        h = (h * 1_000_003 + t + 1) % VOCAB_SIZE
    return h


class OutOfBlocks(Exception):
    """Raised by BlockAllocator.allocate when it cannot produce enough blocks, even after evicting
    every evictable cached block. Nothing about the allocator's state changes when this is raised."""


class Block:
    """One fixed-size slot of KV-cache memory. ref_count is how many sequences currently depend on
    this block's contents. content_hash is set once the block is registered for prefix caching
    (None until then, and again once the block is evicted)."""
    __slots__ = ("block_id", "ref_count", "content_hash")

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.content_hash: Optional[int] = None


class BlockAllocator:
    """Owns all num_blocks physical KV-cache blocks. Every block is in exactly one of three states:
    FREE (ref_count 0, content_hash None), CACHED (ref_count 0, content_hash set -- unused right now
    but still holding valid content a future request could reuse), or LIVE (ref_count >= 1)."""

    def __init__(self, num_blocks: int):
        self.blocks = {i: Block(i) for i in range(num_blocks)}
        self._free = list(range(num_blocks - 1, -1, -1))   # pop() hands out block 0 first
        self._cached_by_hash: dict[int, int] = {}            # content_hash -> block_id, every registered block
        self._lru: "OrderedDict[int, None]" = OrderedDict()  # CACHED block ids, least-recently-freed first

    def num_free_or_evictable(self) -> int:
        """Blocks obtainable without preempting a sequence: the free list plus every CACHED block."""
        return len(self._free) + len(self._lru)

    def allocate(self, count: int) -> list[int]:
        """Returns count fresh block ids with ref_count set to 1, evicting the least-recently-freed
        CACHED blocks first if the free list alone is not enough. Raises OutOfBlocks, with no effect,
        if count > num_free_or_evictable()."""
        if count > self.num_free_or_evictable():
            raise OutOfBlocks(f"need {count}, have {self.num_free_or_evictable()}")
        while len(self._free) < count:
            self._evict_one()
        ids = [self._free.pop() for _ in range(count)]
        for block_id in ids:
            self.blocks[block_id].ref_count = 1
        return ids

    def _evict_one(self) -> None:
        block_id, _ = self._lru.popitem(last=False)
        block = self.blocks[block_id]
        del self._cached_by_hash[block.content_hash]
        block.content_hash = None
        self._free.append(block_id)

    def incref(self, block_id: int) -> None:
        """Adds one reference to a block that is CACHED or already LIVE (removing it from the LRU
        list in the former case -- a block referenced by any sequence is LIVE, never CACHED)."""
        block = self.blocks[block_id]
        if block.ref_count == 0:
            del self._lru[block_id]
        block.ref_count += 1

    def free(self, block_ids: list[int]) -> None:
        """Removes one reference from each block. A block whose ref_count reaches 0 becomes CACHED
        if it has a content_hash, or FREE otherwise."""
        for block_id in block_ids:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                if block.content_hash is None:
                    self._free.append(block_id)
                else:
                    self._lru[block_id] = None   # inserted at the recently-freed end

    def lookup_cached(self, content_hash: int) -> Optional[int]:
        """Returns the id of the CACHED or LIVE block already holding this content_hash, first
        calling incref on it -- or None, with no effect, if no block holds it."""
        block_id = self._cached_by_hash.get(content_hash)
        if block_id is not None:
            self.incref(block_id)
        return block_id

    def register(self, block_id: int, content_hash: int) -> None:
        """Marks the LIVE, not-yet-hashed block block_id as eligible for future reuse under
        content_hash; it stays LIVE (ref_count untouched) until every sequence referencing it frees
        it, at which point it becomes CACHED instead of FREE. Does nothing if content_hash is already
        registered to a different block (two sequences produced the same block content independently;
        the existing entry is kept and block_id stays unhashed)."""
        if content_hash in self._cached_by_hash:
            return
        block = self.blocks[block_id]
        block.content_hash = content_hash
        self._cached_by_hash[content_hash] = block_id


def block_hash(prev_hash: int, block_tokens: list[int]) -> int:
    """Content hash of one full block, chained from the hash of the block before it in the same
    sequence (0 for a sequence's first block). Two blocks hash equal iff every token from the start
    of their sequences through the end of this block is identical."""
    h = prev_hash
    for t in block_tokens:
        h = (h * 1_000_003 + t + 1) & 0xFFFFFFFF
    return h


def full_block_hashes(seq: "Sequence", num_full_blocks: int) -> list[int]:
    """The chained content hash of each of seq's first num_full_blocks full blocks, in order."""
    hashes = []
    h = 0
    for block_index in range(num_full_blocks):
        h = block_hash(h, seq.tokens_in_block(block_index))
        hashes.append(h)
    return hashes


class SeqStatus(Enum):
    WAITING = "waiting"      # queued; prefill has not run (or has not re-run since a preemption)
    RUNNING = "running"      # prefill has run; may still be decoding
    FINISHED = "finished"    # a stop condition has fired; every block has been freed


class Sequence:
    """The engine's-eye view of one request: its prompt, what it has generated, and which physical
    blocks currently hold its KV cache. token_ids is prompt_token_ids + every generated token, in
    order -- the only input next_token needs."""

    def __init__(self, request_id: int, prompt_token_ids: list[int], max_tokens: int):
        self.request_id = request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.max_tokens = max_tokens
        self.token_ids: list[int] = list(prompt_token_ids)
        self.num_generated = 0
        self.status = SeqStatus.WAITING
        self.block_ids: list[int] = []

    @property
    def generated_token_ids(self) -> list[int]:
        return self.token_ids[len(self.prompt_token_ids):]

    def tokens_in_block(self, block_index: int) -> list[int]:
        """The (up to BLOCK_SIZE) tokens belonging to this sequence's block_index-th block."""
        return self.token_ids[block_index * BLOCK_SIZE: (block_index + 1) * BLOCK_SIZE]

    def append_token(self, token_id: int) -> None:
        """Records a newly generated token. Does not touch block_ids or status -- the caller must
        already have made sure a block exists to hold it."""
        self.token_ids.append(token_id)
        self.num_generated += 1

    def is_stopped(self) -> bool:
        """True once max_tokens have been generated, or the last generated token is EOS_TOKEN."""
        return self.num_generated >= self.max_tokens or self.token_ids[-1] == EOS_TOKEN


class RequestQueue:
    """FIFO queue of waiting request ids, with push_front for a request that must be retried before
    anything else waiting."""

    def __init__(self):
        self._dq: deque[int] = deque()

    def push_back(self, request_id: int) -> None:
        self._dq.append(request_id)

    def push_front(self, request_id: int) -> None:
        self._dq.appendleft(request_id)

    def pop_front(self) -> int:
        return self._dq.popleft()

    def peek_front(self) -> Optional[int]:
        return self._dq[0] if self._dq else None

    def __len__(self) -> int:
        return len(self._dq)

class Engine:
    def __init__(self, num_blocks: int):
        self.allocator = BlockAllocator(num_blocks)
        self.queue = RequestQueue()
        self.sequences: dict[int, Sequence] = {}
        self.running: dict[int, Sequence] = {}   # insertion order == admission order, used from Level 5
        self._next_id = 0

    def add_request(self, prompt_token_ids: list[int], max_tokens: int) -> int:
        request_id = self._next_id
        self._next_id += 1
        seq = Sequence(request_id, prompt_token_ids, max_tokens)
        self.sequences[request_id] = seq
        self.queue.push_back(request_id)
        return request_id

    def _prefill(self, seq: Sequence) -> None:
        # NOTE: +1 -- room for the token this call is about to produce, not just for token_ids as it is now
        needed = (len(seq.token_ids) + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
        seq.block_ids.extend(self.allocator.allocate(needed - len(seq.block_ids)))
        seq.append_token(next_token(seq.token_ids))
        seq.status = SeqStatus.RUNNING
        self.running[seq.request_id] = seq

    def step(self) -> None:
        while len(self.queue) > 0:
            self._prefill(self.sequences[self.queue.pop_front()])

class Engine(Engine):
    def __init__(self, num_blocks: int, token_budget: int = 10**9, max_seqs: int = 10**9):
        super().__init__(num_blocks)
        self.token_budget = token_budget
        self.max_seqs = max_seqs

    def _decode_step(self, seq: Sequence) -> None:
        if len(seq.token_ids) > len(seq.block_ids) * BLOCK_SIZE:   # its blocks are overfull
            seq.block_ids.extend(self.allocator.allocate(1))
        seq.append_token(next_token(seq.token_ids))

    def step(self) -> None:
        used = len(self.running)                             # NOTE: one token per running sequence, spent up front
        for request_id in list(self.running):                # NOTE: snapshot -- see Level 5
            seq = self.running.get(request_id)
            if seq is not None:
                self._decode_step(seq)

        while len(self.running) < self.max_seqs and len(self.queue) > 0:
            seq = self.sequences[self.queue.peek_front()]
            if used + len(seq.token_ids) > self.token_budget:   # NOTE: FIFO -- stop, do not skip to a smaller one
                break
            self.queue.pop_front()
            used += len(seq.token_ids)
            self._prefill(seq)

class Engine(Engine):
    def _finish_if_stopped(self, seq: Sequence) -> bool:
        if not seq.is_stopped():
            return False
        self.allocator.free(seq.block_ids)
        seq.block_ids = []
        seq.status = SeqStatus.FINISHED
        self.running.pop(seq.request_id, None)
        return True

    def _prefill(self, seq: Sequence) -> None:
        needed = (len(seq.token_ids) + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
        seq.block_ids.extend(self.allocator.allocate(needed - len(seq.block_ids)))
        seq.append_token(next_token(seq.token_ids))
        if not self._finish_if_stopped(seq):                # NOTE: only reaches RUNNING if it did not just stop
            seq.status = SeqStatus.RUNNING
            self.running[seq.request_id] = seq

    def _decode_step(self, seq: Sequence) -> None:
        if len(seq.token_ids) > len(seq.block_ids) * BLOCK_SIZE:
            seq.block_ids.extend(self.allocator.allocate(1))
        seq.append_token(next_token(seq.token_ids))
        self._finish_if_stopped(seq)

class Engine(Engine):
    def _register_if_newly_full(self, seq: Sequence) -> None:
        if len(seq.token_ids) % BLOCK_SIZE == 0:
            block_index = len(seq.token_ids) // BLOCK_SIZE - 1
            block_hash_ = full_block_hashes(seq, block_index + 1)[-1]
            self.allocator.register(seq.block_ids[block_index], block_hash_)

    def _prefill(self, seq: Sequence) -> None:
        num_full = len(seq.token_ids) // BLOCK_SIZE
        hashes = full_block_hashes(seq, num_full)
        hits = 0
        while hits < num_full:                                # NOTE: stop at the first miss -- see Follow-ups
            cached_id = self.allocator.lookup_cached(hashes[hits])
            if cached_id is None:
                break
            seq.block_ids.append(cached_id)                    # NOTE: lookup_cached already increfs -- no incref here
            hits += 1
        needed = (len(seq.token_ids) + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
        fresh = self.allocator.allocate(needed - hits)
        seq.block_ids.extend(fresh)
        for offset, block_id in enumerate(fresh):
            if hits + offset < num_full:                       # NOTE: never register the trailing partial block
                self.allocator.register(block_id, hashes[hits + offset])
        seq.append_token(next_token(seq.token_ids))
        self._register_if_newly_full(seq)                      # NOTE: the prompt's own tail can complete right here
        if not self._finish_if_stopped(seq):
            seq.status = SeqStatus.RUNNING
            self.running[seq.request_id] = seq

    def _decode_step(self, seq: Sequence) -> None:
        if len(seq.token_ids) > len(seq.block_ids) * BLOCK_SIZE:
            seq.block_ids.extend(self.allocator.allocate(1))
        seq.append_token(next_token(seq.token_ids))
        self._register_if_newly_full(seq)
        self._finish_if_stopped(seq)

class Engine(Engine):
    def _allocate_or_preempt(self, count: int) -> list[int]:
        while True:
            try:
                return self.allocator.allocate(count)
            except OutOfBlocks:
                self._preempt(next(reversed(self.running)))   # NOTE: may be the very sequence asking for room

    def _preempt(self, request_id: int) -> None:
        seq = self.running.pop(request_id)
        self.allocator.free(seq.block_ids)
        seq.block_ids = []
        seq.status = SeqStatus.WAITING
        self.queue.push_front(request_id)

    def _prefill(self, seq: Sequence) -> None:
        num_full = len(seq.token_ids) // BLOCK_SIZE
        hashes = full_block_hashes(seq, num_full)
        hits = 0
        while hits < num_full:
            cached_id = self.allocator.lookup_cached(hashes[hits])
            if cached_id is None:
                break
            seq.block_ids.append(cached_id)
            hits += 1
        needed = (len(seq.token_ids) + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
        fresh = self._allocate_or_preempt(needed - hits)   # NOTE: seq is never its own victim here -- it is not
        seq.block_ids.extend(fresh)                          #  in self.running until this call returns
        for offset, block_id in enumerate(fresh):
            if hits + offset < num_full:
                self.allocator.register(block_id, hashes[hits + offset])
        seq.append_token(next_token(seq.token_ids))
        self._register_if_newly_full(seq)
        if not self._finish_if_stopped(seq):
            seq.status = SeqStatus.RUNNING
            self.running[seq.request_id] = seq

    def _decode_step(self, seq: Sequence) -> None:
        if len(seq.token_ids) > len(seq.block_ids) * BLOCK_SIZE:
            fresh = self._allocate_or_preempt(1)
            if seq.request_id not in self.running:      # NOTE: seq preempted itself while making its own room
                self.allocator.free(fresh)                #  -- it is WAITING now; hand the block straight back
                return
            seq.block_ids.append(fresh[0])
        seq.append_token(next_token(seq.token_ids))
        self._register_if_newly_full(seq)
        self._finish_if_stopped(seq)
