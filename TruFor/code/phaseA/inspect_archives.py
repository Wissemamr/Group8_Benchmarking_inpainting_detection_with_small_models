"""Inspect tar.gz archives: member types, safety (abs paths, '..', links), roots, categories, sizes."""
import sys, tarfile, collections
for path in sys.argv[1:]:
    roots, cats, kinds, bad = collections.Counter(), set(), collections.Counter(), []
    size = 0
    with tarfile.open(path, "r:gz") as t:
        for m in t:
            kinds["dir" if m.isdir() else "file" if m.isfile() else "other"] += 1
            n = m.name
            if n.startswith("/") or ".." in n.split("/") or not (m.isdir() or m.isfile()):
                bad.append(n)
            parts = n.split("/")
            roots[parts[0]] += 1
            if len(parts) >= 2 and parts[1]:
                cats.add(parts[1])
            size += m.size
    print(f"== {path}\n  members={dict(kinds)} uncompressed={size/1e9:.3f} GB")
    print(f"  top-level={dict(roots)}  categories={len(cats)}  unsafe/odd members={len(bad)} {bad[:5]}")
    print(f"  categories sample: {sorted(cats)[:8]}")
