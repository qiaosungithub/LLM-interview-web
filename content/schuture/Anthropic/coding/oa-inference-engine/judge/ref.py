"""A literal model of the five levels, used for the expected values of the random tests, and the random workload
generator. The statement's given code comes first, copied unchanged: the model uses that BlockAllocator, so its
block ids are the ones a correct answer produces."""

# ---------------------------------------------------------------- the statement's given code (unchanged)

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


# ---------------------------------------------------------------- a literal model of the five levels

class Invalid(Exception):
    """The workload leaves the statement's guarantees: the blocks ran out before Level 5."""


class Req:
    """One request as the model sees it."""
    __slots__ = ("prompt", "tokens", "max_tokens", "status", "blocks")

    def __init__(self, prompt, max_tokens):
        self.prompt = list(prompt)
        self.tokens = list(prompt)
        self.max_tokens = max_tokens
        self.status = "WAITING"
        self.blocks = []

    def stopped(self):
        return len(self.tokens) - len(self.prompt) >= self.max_tokens or self.tokens[-1] == EOS_TOKEN


def chain_hashes(tokens, count):
    """The content hash of each of the first count full blocks of tokens."""
    out, h = [], 0
    for i in range(count):
        h = block_hash(h, tokens[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE])
        out.append(h)
    return out


class Model:
    """The engine after a given level, following the statement's rules one by one."""

    def __init__(self, level, num_blocks, token_budget=None, max_seqs=None):
        self.level = level
        self.alloc = BlockAllocator(num_blocks)
        self.budget = 10**9 if token_budget is None else token_budget
        self.max_seqs = 10**9 if max_seqs is None else max_seqs
        self.reqs = []
        self.waiting = deque()      # request ids, front first
        self.running = []           # request ids in admission order, the latest admitted last
        self.open_point = False     # a request stopped on the very token that filled a block (Level 4+): the
        #                             statement does not say whether that block is registered before it is freed

    def add_request(self, prompt, max_tokens):
        self.reqs.append(Req(prompt, max_tokens))
        self.waiting.append(len(self.reqs) - 1)
        return len(self.reqs) - 1

    def step(self):
        if self.level == 1:
            while self.waiting:
                self._prefill(self.waiting.popleft())
            return
        spent = len(self.running)                      # one token for every sequence RUNNING as the step starts
        for rid in list(self.running):
            if self.reqs[rid].status == "RUNNING":     # skipped if preempted earlier in this loop
                self._decode(rid)
        while len(self.running) < self.max_seqs and self.waiting:
            head = self.reqs[self.waiting[0]]
            if spent + len(head.tokens) > self.budget:
                break
            spent += len(head.tokens)
            self._prefill(self.waiting.popleft())

    # ---- the pieces

    def _reserve(self, count):
        while count > self.alloc.num_free_or_evictable():
            if self.level < 5 or not self.running:
                raise Invalid
            self._preempt(self.running[-1])
        return self.alloc.allocate(count)

    def _prefill(self, rid):
        q = self.reqs[rid]
        hits, hashes = [], []
        if self.level >= 4:
            hashes = chain_hashes(q.tokens, len(q.tokens) // BLOCK_SIZE)
            for h in hashes:
                block_id = self.alloc.lookup_cached(h)
                if block_id is None:
                    break
                hits.append(block_id)
        need = -(-(len(q.tokens) + 1) // BLOCK_SIZE)
        q.blocks = hits + self._reserve(need - len(hits))
        for i in range(len(hits), len(hashes)):
            self.alloc.register(q.blocks[i], hashes[i])
        q.status = "RUNNING"
        self.running.append(rid)
        self._append(rid)

    def _decode(self, rid):
        q = self.reqs[rid]
        need = -(-(len(q.tokens) + 1) // BLOCK_SIZE) - len(q.blocks)
        if need > 0:
            fresh = self._reserve(need)
            if q.status != "RUNNING":                  # it was the latest admitted and preempted itself
                self.alloc.free(fresh)
                return
            q.blocks += fresh
        self._append(rid)

    def _append(self, rid):
        q = self.reqs[rid]
        q.tokens.append(next_token(q.tokens))
        filled = len(q.tokens) % BLOCK_SIZE == 0
        if self.level >= 4 and filled:
            n = len(q.tokens) // BLOCK_SIZE
            self.alloc.register(q.blocks[n - 1], chain_hashes(q.tokens, n)[-1])
        if self.level >= 3 and q.stopped():
            self.open_point |= self.level >= 4 and filled
            self.alloc.free(q.blocks)
            q.blocks = []
            q.status = "FINISHED"
            self.running.remove(rid)

    def _preempt(self, rid):
        q = self.reqs[rid]
        self.running.remove(rid)
        self.alloc.free(q.blocks)
        q.blocks = []
        q.status = "WAITING"
        self.waiting.appendleft(rid)

    # ---- what the tests compare after every step

    def observe(self, level):
        """(probe, value) pairs; probes are ("gen", rid), ("status", rid), ("blocks", rid), ("free",), and from
        Level 4 on ("refs",) and ("cached",)."""
        out = []
        for rid, q in enumerate(self.reqs):
            out += [(("gen", rid), q.tokens[len(q.prompt):]), (("status", rid), q.status),
                    (("blocks", rid), list(q.blocks))]
        out.append((("free",), self.alloc.num_free_or_evictable()))
        if level >= 4:
            out.append((("refs",), [b.ref_count for b in self.alloc.blocks.values()]))
            out.append((("cached",), [i for i, b in self.alloc.blocks.items()
                                      if b.ref_count == 0 and b.content_hash is not None]))
        return out


def trace(level, config, ops, probes):
    """Runs ops ("add", prompt, max_tokens) / ("step",) on Model(level); per op the add's request id or the
    observation after the step (probes of level `probes`). Returns (results, model)."""
    m = Model(level, **config)
    out = []
    for op in ops:
        if op[0] == "add":
            out.append(m.add_request(op[1], op[2]))
        else:
            m.step()
            out.append(m.observe(probes))
    return out, m


def expected(level, config, ops):
    """What a correct final Engine shows for ops in a test of `level`, or None when ops is not a fair test of
    that level: the blocks run out before Level 5, a request stops on a block-filling token at Level 4+ (the
    open point above), or a later level's rules would change what this level's rules say (a Level 2 sequence
    that would already stop, two prompts that would share a cached block before Level 4, ...)."""
    try:
        final, m = trace(5, config, ops, level)
        if m.open_point:
            return None
        if level < 5 and trace(level, config, ops, level)[0] != final:
            return None
    except Invalid:
        return None
    return final


# ---------------------------------------------------------------- random workloads

def naive_generate(prompt, max_tokens):
    """What one request generates on its own."""
    tokens, out = list(prompt), []
    while len(out) < max_tokens and (not out or out[-1] != EOS_TOKEN):
        t = next_token(tokens)
        tokens.append(t)
        out.append(t)
    return out


def _blocks_for(n_tokens):
    return -(-n_tokens // BLOCK_SIZE)


def _tokens(r, n):
    return [r.randrange(VOCAB_SIZE) for _ in range(n)]


def _request(r, level, made):
    """A prompt and max_tokens for one request; from Level 4 on, often sharing a prefix with an earlier request
    (its prompt or what it will have generated), so that cache hits happen. Never one whose final length is a
    multiple of BLOCK_SIZE (the open point in Model)."""
    while True:
        if level <= 2:
            prompt, max_tokens = _tokens(r, r.randint(1, 8)), r.randint(8, 12)
            if EOS_TOKEN in naive_generate(prompt, 8):
                continue                                   # it would stop within the steps a Level 2 test runs
            return prompt, max_tokens
        # Level 5 runs longer requests from shorter prompts, so that sequences grow into each other's room
        max_tokens = r.randint(3, 9) if level == 5 else r.randint(1, 6)
        longest = 7 if level == 5 else 10
        if level >= 4 and made and r.random() < (0.55 if level == 5 else 0.7):
            p, mt = r.choice(made)
            full = p + naive_generate(p, mt)
            cut = r.choice([4, 4, 8, len(p), len(full), r.randint(1, len(full))])
            prompt = full[:cut] + _tokens(r, r.choice([0, 0, 1, 2, 3, 4, 5]))
            prompt = prompt[:longest] or _tokens(r, 1)
        else:
            prompt = _tokens(r, r.randint(1, longest - 1))
        if (len(prompt) + len(naive_generate(prompt, max_tokens))) % BLOCK_SIZE:
            return prompt, max_tokens


def _setup(r, level):
    """The engine's arguments and the requests (prompt, max_tokens) of one random test of level."""
    reqs = []
    for _ in range(r.randint(1, 6) if level == 1 else r.randint(3, 8) if level == 5 else r.randint(2, 8)):
        reqs.append(_request(r, level, reqs))
    max_need = max(_blocks_for(len(p) + mt) for p, mt in reqs)
    if level == 1:
        total = sum(_blocks_for(len(p) + 1) for p, _ in reqs)
        return {"num_blocks": max(max_need, total) + r.randint(0, 3)}, reqs
    max_seqs = r.randint(2, 5) if level == 5 else r.randint(1, 4)
    if level == 5:      # few spare blocks: preemption happens often
        budget = max(max(len(p) + mt for p, mt in reqs), max_seqs) + r.randint(0, 8)
        num_blocks = max_need + r.choice([0, 1, 1, 2])
    else:
        budget = max(max(len(p) for p, _ in reqs), max_seqs) + r.randint(0, 8)
        total = sum(_blocks_for(len(p) + mt) for p, mt in reqs)
        num_blocks = r.randint(max_need, max(max_need, total))
    if level == 2:      # a Level 2 test runs at most 7 steps and every sequence it admits keeps running
        num_blocks = max(num_blocks, sum(_blocks_for(len(p) + 8) for p, _ in reqs))
    return {"num_blocks": num_blocks, "token_budget": budget, "max_seqs": max_seqs}, reqs


def random_workload(r, level):
    """(config, ops, want) for a fair random test of level: want holds, per op, what a correct final Engine
    shows. The requests arrive in bursts between steps. The final engine runs side by side with the engine of
    this level, and the test ends before the first step where they would differ (or where the test would leave
    the statement's guarantees), so every expectation is one this level's rules already fix."""
    while True:
        config, reqs = _setup(r, level)
        ops, want = _drive(r, level, config, reqs)
        if sum(op[0] == "step" for op in ops) >= (1 if level == 1 else 2):
            return config, ops, want


def _drive(r, level, config, reqs):
    final = Model(5, **config)
    models = [final] if level == 5 else [Model(level, **config), final]
    ops, want, pending = [], [], list(reqs)

    def arrive(k):
        for p, mt in pending[:k]:
            ops.append(("add", p, mt))
            want.append([m.add_request(p, mt) for m in models][-1])
        del pending[:k]

    if level == 1:
        if r.random() < 0.15:                   # a step before any request arrives
            for m in models:
                m.step()
            ops.append(("step",))
            want.append(final.observe(level))
        arrive(len(pending))
        limit = 1
    else:
        arrive(r.randint(1, len(pending)))
        limit = r.randint(2, 7) if level == 2 else 80
    steps = 0
    while steps < limit:
        try:
            for m in models:
                m.step()
        except Invalid:
            break
        seen = [m.observe(level) for m in models]
        if final.open_point or seen[0] != seen[-1]:
            break
        ops.append(("step",))
        want.append(seen[-1])
        steps += 1
        if pending and r.random() < 0.4:
            arrive(r.randint(1, len(pending)))
        elif not pending and not final.running and not final.waiting:
            if r.random() < 0.3:
                ops.append(("step",))           # a step with nothing to do
                want.append(final.observe(level))
            break
    while ops and ops[-1][0] == "add":
        ops.pop()
        want.pop()
    return ops, want
