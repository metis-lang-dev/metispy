"""A list kept as a window over a longer history, indexed by absolute
position (watch-history s-e061fad2).

A long-running watch fires forever: the REPL's per-row history (fired,
fired_rows, fired_how, the env snapshots, the ledger) would grow without
bound. A Window keeps only its last items but answers every position as
the whole history numbers it: len() is the absolute length (everything
ever appended), w[k] is the item at absolute position k, and so every
checkpoint taken as len(...) keeps its meaning across a trim.

    w = Window(); w.append(...)      # as a list
    w.trim(n)                        # drop the first n items still kept
    w.off                            # the absolute position of the first one kept

Reads before the window: an index raises Trimmed (an IndexError); a slice
starts at the window (what is kept of that range). A cut before it
(del w[k:] with k < off) raises Trimmed: there is nothing there to cut back
to. Iteration yields the kept items only."""


class Trimmed(IndexError):
    """A position before the window: the history kept starts later."""


class Window:
    __slots__ = ("items", "off")

    def __init__(self, items=(), off=0):
        self.items = list(items)
        self.off = off

    # -- the absolute view ---------------------------------------------------
    def __len__(self):
        return self.off + len(self.items)

    def __bool__(self):
        return len(self) > 0

    def _at(self, k):
        n = len(self)
        if k < 0:
            k += n
        if k < 0 or k >= n:
            raise IndexError(f"position {k} out of 0..{n - 1}")
        if k < self.off:
            raise Trimmed(f"position {k} is before the history kept (#{self.off}..)")
        return k - self.off

    def _span(self, s, cut=False):
        start, stop, step = s.indices(len(self))
        if step != 1:
            raise ValueError("a Window slices with step 1 only")
        if start < self.off:
            if cut:
                raise Trimmed(f"position {start} is before the history kept (#{self.off}..)")
            start = self.off
        stop = max(stop, start)
        return start - self.off, stop - self.off

    def __getitem__(self, k):
        if isinstance(k, slice):
            a, b = self._span(k)
            return self.items[a:b]
        return self.items[self._at(k)]

    def __setitem__(self, k, v):
        if isinstance(k, slice):
            raise TypeError("a Window assigns one position at a time")
        self.items[self._at(k)] = v

    def __delitem__(self, k):
        if not isinstance(k, slice) or k.stop is not None:
            raise TypeError("a Window cuts its tail only (del w[k:])")
        a, _ = self._span(k, cut=True)
        del self.items[a:]

    def __iter__(self):
        return iter(self.items)

    def __reversed__(self):
        return reversed(self.items)

    def __eq__(self, other):
        if isinstance(other, Window):
            return self.off == other.off and self.items == other.items
        return self.off == 0 and self.items == list(other)

    def __repr__(self):
        return f"Window(off={self.off}, {self.items!r})"

    # -- the list verbs the REPL uses -----------------------------------------
    def append(self, v):
        self.items.append(v)

    def extend(self, vs):
        self.items.extend(vs)

    def count(self, v):
        """In the items kept."""
        return self.items.count(v)

    def index(self, v):
        """The absolute position of v's first occurrence kept."""
        return self.off + self.items.index(v)

    def __contains__(self, v):
        return v in self.items

    def pop(self, i=-1):
        if i != -1:
            raise TypeError("a Window pops its last item only")
        if not self.items:
            raise Trimmed("the history kept is empty") if self.off else IndexError("pop from an empty Window")
        return self.items.pop()

    # -- the window ------------------------------------------------------------
    def trim(self, n):
        """Drop the first n items kept (the oldest): -> how many went."""
        n = max(0, min(n, len(self.items)))
        if n:
            del self.items[:n]
            self.off += n
        return n

    def kept(self):
        """The absolute positions kept: range(off, len)."""
        return range(self.off, len(self))
