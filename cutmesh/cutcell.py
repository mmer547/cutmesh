"""Cut-cell kernel: split general polyhedra by a level set given at vertices.

A cell is a list of (face_id, polygon) with polygons as lists of global vertex ids,
oriented outward.  Splitting is decided purely from the vertex values, and crossing
vertices are keyed by their (sorted) edge, so neighbouring cells always split a
shared face identically -> the result is conformal by construction.

Vertex classification: N if phi <= 0, P if phi > 0.  A vertex with phi == 0 (snapped
onto the surface) is used directly as the crossing point, so no slivers are created.
"""


def _dedupe(p):
    out = []
    for v in p:
        if not out or out[-1] != v:
            out.append(v)
    while len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


class CutContext:
    def __init__(self, positions, phi, next_face_id):
        self.pos = positions          # python list of [x,y,z] (grows)
        self.phi = phi                # python list (grows) - current level set
        self.cache = {}
        self.new_vertices = []        # ids created in this pass
        self.new_parents = []         # (lo, hi) edge each new vertex came from
        self.subids = {}
        self.next_id = next_face_id
        self.chord_hook = None        # (ctx, poly, a, b) -> [x,y,z] feature point on chord a-b, or None
        self.chord_cache = {}
        self.onfeat = set()           # vertices lying on the feature edges of this pass
        self.used_chords = []         # chord_cache keys with a feature point, used by cut()
        self.zero_edge = {}           # (face, lo, hi) -> +1 solid / -1 fluid for edges between phi == 0 vertices
        self.pinch_hook = None        # (ctx, poly, a, b, candidates) -> surface vertex the chord a-b bends through, or None
        self.pinch_cache = {}
        self.pinch_min_area = 0.05    # each kept piece of a bent face, as a fraction of the kept part
        self.pinch_now = set()        # vertices the chords of the current cut bend through
        self.last_pinch = set()       # of the last cut: the line the kept side is pinched along

    # ------------------------------------------------------------------
    def new_face_id(self):
        i = self.next_id
        self.next_id += 1
        return i

    def sub_id(self, key):
        i = self.subids.get(key)
        if i is None:
            i = self.new_face_id()
            self.subids[key] = i
        return i

    def cross(self, a, b):
        lo, hi = (a, b) if a < b else (b, a)
        key = (lo, hi)
        c = self.cache.get(key)
        if c is not None:
            return c
        pl, ph = self.phi[lo], self.phi[hi]
        if pl == 0.0:
            c = lo
        elif ph == 0.0:
            c = hi
        else:
            t = pl / (pl - ph)
            A, B = self.pos[lo], self.pos[hi]
            c = len(self.pos)
            self.pos.append([A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1]),
                             A[2] + t * (B[2] - A[2])])
            self.phi.append(0.0)
            self.new_vertices.append(c)
            self.new_parents.append(key)
        self.cache[key] = c
        return c

    def chord_point(self, fid, poly, a, b):
        """Feature point where a sharp edge pierces the face, inserted into the chord
        a-b.  Cached per (face, chord), so both cells sharing the face agree."""
        if a == b:
            return None
        key = (fid, a, b) if a < b else (fid, b, a)
        if key in self.chord_cache:
            c = self.chord_cache[key]
            if c is not None:
                self.used_chords.append(key)
            return c
        x = self.chord_hook(self, poly, a, b)
        c = None
        if x is not None:
            c = len(self.pos)
            self.pos.append(list(x))
            self.phi.append(0.0)
            self.new_vertices.append(c)
            self.new_parents.append((-1, -1))
            self.onfeat.add(c)
            self.used_chords.append(key)
        self.chord_cache[key] = c
        return c

    def _zero_runs(self, fid, poly, cls):
        """Runs of phi == 0 vertices between an N and a P vertex (surface vertices at a
        concave edge, e.g. on a wall plane next to a hole): the chord must end where
        the face leaves the fluid, not at the last vertex of the run.  The edges of the
        run are classified per face (self.zero_edge), so both cells agree."""
        phi = self.phi
        n = len(poly)
        z = [phi[v] == 0.0 for v in poly]
        for i in range(n):
            if not z[i] or z[i - 1]:
                continue
            k = 0
            while k < n and z[(i + k) % n]:
                k += 1
            if k < 2 or k == n:
                continue
            run = [(i + t) % n for t in range(k)]
            pn = phi[poly[i - 1]] < 0.0
            if pn == (phi[poly[(i + k) % n]] < 0.0):
                continue
            if not pn:
                run = run[::-1]               # walk from the N side
            for t in range(k - 1):
                a, b = poly[run[t]], poly[run[t + 1]]
                key = (fid, a, b) if a < b else (fid, b, a)
                e = self.zero_edge.get(key)
                if e == 1:
                    for r in run[t + 1:]:
                        cls[r] = False
                if e != -1:
                    break                     # solid found, or unknown: stop
        return cls

    def _xing(self, a, b, a_neg):
        """crossing on edge a-b; between two surface vertices it is the N one"""
        if self.phi[a] == 0.0 and self.phi[b] == 0.0:
            return a if a_neg else b
        return self.cross(a, b)

    def forbid_chord(self, key):
        """fall back to the straight chord on this face (both cells must be re-cut)"""
        self.chord_cache[key] = None
        self.pinch_cache[key] = None

    def pinch_point(self, fid, poly, npoly, a, b):
        """Surface vertex of the kept part of the face beyond its chord a-b (the tip of a
        wedge of solid at a concave edge, the vertex on the edge): the chord bends
        through it, which splits the kept part in two.  Cached per (face, chord)."""
        key = (fid, a, b) if a < b else (fid, b, a)
        if key in self.pinch_cache:
            c = self.pinch_cache[key]
        else:
            n = len(npoly)
            ia, ib = npoly.index(a), npoly.index(b)
            near = {a, b, npoly[ia - 1], npoly[(ib + 1) % n]}
            cand = [v for v in npoly if self.phi[v] == 0.0 and v not in near]
            c = self.pinch_hook(self, poly, a, b, cand) if cand else None
            self.pinch_cache[key] = c
        if c is not None:
            self.used_chords.append(key)
        return c

    # ------------------------------------------------------------------
    def classify(self, faces):
        """-1: entirely N side, +1: entirely P side, 0: needs cutting."""
        phi = self.phi
        hasN = hasP = False
        for _, poly in faces:
            for v in poly:
                x = phi[v]
                if x < 0.0:
                    hasN = True
                elif x > 0.0:
                    hasP = True
            if hasN and hasP:
                return 0
        if hasP:
            return 1
        return -1

    def cut(self, faces, shared_cap, cap_hook=None):
        """Split a cell. Returns (neg_faces, pos_faces, cap_polys).

        Face assignment depends only on the face itself (so both cells sharing it agree):
        no strictly-positive vertex -> whole face on N side; no strictly-negative vertex
        -> whole face on P side; otherwise split along the crossing chord(s).  The cap
        is the boundary of the N-side face set, which makes it consistent by construction.

        shared_cap=True: cap polygons get one face id used by both sides (an internal
        interface face); otherwise only the neg side's cap is meaningful.
        """
        phi = self.phi
        neg, pos = [], []
        self.used_chords = []
        self.pinch_now = set()
        for fid, poly in faces:
            hasN = hasP = False
            for v in poly:
                x = phi[v]
                if x < 0.0:
                    hasN = True
                elif x > 0.0:
                    hasP = True
            if not hasP:
                neg.append((fid, poly)); continue
            if not hasN:
                pos.append((fid, poly)); continue
            n = len(poly)
            cls = [phi[v] <= 0.0 for v in poly]
            if self.zero_edge:
                cls = self._zero_runs(fid, poly, cls)
            s = 0
            for i in range(n):
                if cls[i] and not cls[i - 1]:
                    s = i; break
            seqN, ppieces, cur = [], [], None
            chords = []
            for step in range(n):
                i = (s + step) % n
                j = (i + 1) % n
                a = poly[i]
                if cls[i]:
                    seqN.append(a)
                else:
                    cur.append(a)
                if cls[i] and not cls[j]:
                    c = self._xing(a, poly[j], True)
                    seqN.append(c)
                    cur = [c]
                elif (not cls[i]) and cls[j]:
                    c = self._xing(a, poly[j], False)
                    f = self.chord_point(fid, poly, cur[0], c) if self.chord_hook else None
                    chords.append((cur[0], c, f))
                    cur.append(c)
                    if f is not None:          # chord c_out -> f -> c_in follows the edge
                        cur.append(f)
                        seqN.append(f)
                    ppieces.append(cur)
                    seqN.append(c)
                    cur = None
            pieces = []
            npoly = _dedupe(seqN)
            T = None
            if self.pinch_hook is not None and len(chords) == 1 and chords[0][2] is None                     and len(set(npoly)) == len(npoly) >= 4:
                a, b, _ = chords[0]
                if a != b and a in npoly and b in npoly and                         npoly[(npoly.index(b) - 1) % len(npoly)] == a:
                    T = self.pinch_point(fid, poly, npoly, a, b)
            if T is not None:
                # kept part: [T, b, ...] and [T, ..., a]; the solid part gets T
                ib = npoly.index(b)
                seq = npoly[ib:] + npoly[:ib]
                it = seq.index(T)
                k1, k2 = [T] + seq[:it], seq[it:]
                # both parts real cells, not slivers along a wall
                a1, a2, a0 = self._area(k1), self._area(k2), self._area(npoly)
                if min(a1, a2) < self.pinch_min_area * a0:
                    T = None
            if T is not None:
                pieces.append((0, k1))
                pieces.append((0, k2))
                ppieces[0].append(T)
                self.pinch_now.add(T)
            elif len(set(npoly)) >= 3:
                pieces.append((0, npoly))
            for pp in ppieces:
                pp = _dedupe(pp)
                if len(set(pp)) >= 3:
                    pieces.append((1, pp))
            if len(pieces) == 1:
                (neg if pieces[0][0] == 0 else pos).append((fid, poly))
                continue
            for k, pp in pieces:
                # key by vertex set: independent of traversal direction / start
                fid2 = self.sub_id((fid, tuple(sorted(pp))))
                (neg if k == 0 else pos).append((fid2, pp))
        # cap = reversed open boundary of the N-side faces (of each part, where the
        # kept side is pinched along a line of surface vertices)
        self.last_pinch = self._pinch_line(neg) if len(self.pinch_now) >= 2 else set()
        groups = split_components(neg, self.last_pinch) if self.last_pinch else [neg]
        caps = []
        for grp in groups:
            de = {}
            for _, p in grp:
                m = len(p)
                for k in range(m):
                    e = (p[k], p[k + 1 if k + 1 < m else 0])
                    de[e] = de.get(e, 0) + 1
            segs = []
            for (a, b), c in de.items():
                r = de.get((b, a), 0)
                for _ in range(c - r):
                    segs.append((b, a))
            caps += _chain_loops(segs)
        out_caps = []
        for loop in caps:
            polys = cap_hook(self, loop) if cap_hook is not None else [loop]
            for poly in polys:
                out_caps.append(poly)
                if shared_cap:
                    cid = self.new_face_id()
                    neg.append((cid, poly))
                    pos.append((cid, poly[::-1]))
                else:
                    neg.append((self.new_face_id(), poly))
                    pos.append((self.new_face_id(), poly[::-1]))
        return neg, pos, out_caps

    def _area(self, poly):
        pos = self.pos
        x0 = pos[poly[0]]
        s = [0.0, 0.0, 0.0]
        for k in range(1, len(poly) - 1):
            u = [pos[poly[k]][i] - x0[i] for i in range(3)]
            w = [pos[poly[k + 1]][i] - x0[i] for i in range(3)]
            s[0] += u[1] * w[2] - u[2] * w[1]
            s[1] += u[2] * w[0] - u[0] * w[2]
            s[2] += u[0] * w[1] - u[1] * w[0]
        return 0.5 * (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]) ** 0.5

    def _pinch_line(self, neg):
        """the bend vertices of this cut and the surface vertices of the kept faces in line
        between two of them (e.g. hanging vertices on the edge)"""
        pv = sorted(self.pinch_now)
        pos = self.pos
        line = set(pv)
        zs = set(v for _, p in neg for v in p if self.phi[v] == 0.0) - line
        for i in range(len(pv)):
            A = pos[pv[i]]
            for j in range(i + 1, len(pv)):
                B = pos[pv[j]]
                u = [B[0] - A[0], B[1] - A[1], B[2] - A[2]]
                L2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2]
                if L2 == 0.0:
                    continue
                for v in zs:
                    X = pos[v]
                    w = [X[0] - A[0], X[1] - A[1], X[2] - A[2]]
                    t = (w[0] * u[0] + w[1] * u[1] + w[2] * u[2]) / L2
                    if not 0.0 < t < 1.0:
                        continue
                    d = [w[0] - t * u[0], w[1] - t * u[1], w[2] - t * u[2]]
                    if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < 0.01 * L2:
                        line.add(v)
        return line


def _chain_loops(segs):
    """Decompose directed segments (each vertex in-degree == out-degree) into loops."""
    out = {}
    for a, b in segs:
        out.setdefault(a, []).append(b)
    loops = []
    for start in list(out.keys()):
        while out.get(start):
            loop = [start]
            v = out[start].pop()
            guard = 0
            while v != start and guard < 100000:
                loop.append(v)
                nxt = out.get(v)
                if not nxt:
                    break
                v = nxt.pop()
                guard += 1
            if v == start:
                loops.extend(_simple_loops(loop))
    return loops


def _simple_loops(loop):
    """Split a closed loop that passes a vertex more than once (two cap loops touching
    at a vertex, chained into a figure eight) into simple loops; drop degenerate ones."""
    out, stack, pos = [], [], {}
    for v in loop:
        i = pos.get(v)
        if i is None:
            pos[v] = len(stack)
            stack.append(v)
            continue
        sub = stack[i:]                  # v ... back to v
        out.append(sub)
        for u in sub[1:]:
            del pos[u]
        del stack[i + 1:]
    out.append(stack)
    return [s for s in out if len(s) >= 3]


def split_components(faces, apart=None):
    """Split a face list into edge-connected components (separate cells).  apart:
    vertices along which the faces are not joined (a kept side pinched along a line)."""
    n = len(faces)
    if n <= 4:
        return [faces]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}
    for i, (_, p) in enumerate(faces):
        m = len(p)
        for k in range(m):
            a = p[k]; b = p[k + 1 if k + 1 < m else 0]
            if apart and a in apart and b in apart:
                continue
            key = (a, b) if a < b else (b, a)
            j = owner.get(key)
            if j is None:
                owner[key] = i
            else:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(faces[i])
    return list(groups.values())
