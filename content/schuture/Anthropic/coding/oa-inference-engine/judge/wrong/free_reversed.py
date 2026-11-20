# expect: WA
# A sequence's blocks are freed last block first, so they reach the free list / LRU order in reverse.
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
    def __init__(self, num_blocks: int, token_budget: int = 10**9, max_seqs: int = 10**9) -> None:
        self.allocator = BlockAllocator(num_blocks)
        self.sequences: dict[int, Sequence] = {}
        self.queue = RequestQueue()
        self.token_budget = token_budget
        self.max_seqs = max_seqs
        self.active: list[int] = []       # RUNNING request ids, oldest admission first

    def add_request(self, prompt_token_ids: list[int], max_tokens: int) -> int:
        rid = len(self.sequences)
        self.sequences[rid] = Sequence(rid, prompt_token_ids, max_tokens)
        self.queue.push_back(rid)
        return rid

    def _release(self, seq: Sequence) -> None:
        self.allocator.free(seq.block_ids[::-1])
        seq.block_ids = []

    def _grab(self, count: int) -> list[int]:
        """count fresh blocks in one allocate call, preempting the latest admitted RUNNING sequence until it fits."""
        while True:
            try:
                return self.allocator.allocate(count)
            except OutOfBlocks:
                victim = self.active.pop()
                vs = self.sequences[victim]
                self._release(vs)
                vs.status = SeqStatus.WAITING
                self.queue.push_front(victim)

    def _after_append(self, seq: Sequence) -> bool:
        """Registers a block the new token just filled; stops the sequence if it should. True if it stopped."""
        n = len(seq.token_ids)
        if n % BLOCK_SIZE == 0:
            k = n // BLOCK_SIZE
            self.allocator.register(seq.block_ids[k - 1], full_block_hashes(seq, k)[-1])
        if seq.is_stopped():
            self._release(seq)
            seq.status = SeqStatus.FINISHED
            if seq.request_id in self.active:
                self.active.remove(seq.request_id)
            return True
        return False

    def _prefill(self, rid: int) -> None:
        seq = self.sequences[rid]
        total = -(-(len(seq.token_ids) + 1) // BLOCK_SIZE)
        hashes = full_block_hashes(seq, len(seq.token_ids) // BLOCK_SIZE)
        hits: list[int] = []
        for h in hashes:
            b = self.allocator.lookup_cached(h)
            if b is None:
                break
            hits.append(b)
        fresh = self._grab(total - len(hits))
        seq.block_ids = hits + fresh
        for i in range(len(hits), len(hashes)):
            self.allocator.register(seq.block_ids[i], hashes[i])
        seq.append_token(next_token(seq.token_ids))
        if not self._after_append(seq):
            seq.status = SeqStatus.RUNNING
            self.active.append(rid)

    def _decode(self, rid: int) -> None:
        seq = self.sequences[rid]
        if len(seq.token_ids) % BLOCK_SIZE == 0:
            fresh = self._grab(1)
            if seq.status is not SeqStatus.RUNNING:     # it was the latest admitted and preempted itself
                self.allocator.free(fresh)
                return
            seq.block_ids += fresh
        seq.append_token(next_token(seq.token_ids))
        self._after_append(seq)

    def step(self) -> None:
        cost = len(self.active)
        for rid in list(self.active):
            if self.sequences[rid].status is SeqStatus.RUNNING:
                self._decode(rid)
        while len(self.active) < self.max_seqs and len(self.queue):
            nxt = self.sequences[self.queue.peek_front()]
            if cost + len(nxt.token_ids) > self.token_budget:
                break
            cost += len(nxt.token_ids)
            self._prefill(self.queue.pop_front())
