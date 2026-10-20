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

    def forbid_chord(self, key):
        """fall back to the straight chord on this face (both cells must be re-cut)"""
        self.chord_cache[key] = None

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
            s = 0
            for i in range(n):
                if cls[i] and not cls[i - 1]:
                    s = i; break
            seqN, ppieces, cur = [], [], None
            for step in range(n):
                i = (s + step) % n
                j = (i + 1) % n
                a = poly[i]
                if cls[i]:
                    seqN.append(a)
                else:
                    cur.append(a)
                if cls[i] and not cls[j]:
                    c = self.cross(a, poly[j])
                    seqN.append(c)
                    cur = [c]
                elif (not cls[i]) and cls[j]:
                    c = self.cross(a, poly[j])
                    f = self.chord_point(fid, poly, cur[0], c) if self.chord_hook else None
                    cur.append(c)
                    if f is not None:          # chord c_out -> f -> c_in follows the edge
                        cur.append(f)
                        seqN.append(f)
                    ppieces.append(cur)
                    seqN.append(c)
                    cur = None
            pieces = []
            npoly = _dedupe(seqN)
            if len(set(npoly)) >= 3:
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
        # cap = reversed open boundary of the N-side faces
        de = {}
        for _, p in neg:
            m = len(p)
            for k in range(m):
                e = (p[k], p[k + 1 if k + 1 < m else 0])
                de[e] = de.get(e, 0) + 1
        segs = []
        for (a, b), c in de.items():
            r = de.get((b, a), 0)
            for _ in range(c - r):
                segs.append((b, a))
        caps = _chain_loops(segs)
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
            if v == start and len(set(loop)) >= 3:
                loops.append(loop)
    return loops


def split_components(faces):
    """Split a face list into edge-connected components (separate cells)."""
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
