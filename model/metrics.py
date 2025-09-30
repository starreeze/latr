from numba import jit


@jit(nopython=True)
def longest_prefix_in_text(pattern: list[int], text: list[int]) -> int:
    """
    Return the maximum length L such that pattern[:L] occurs as a contiguous
    subsequence in text. Runs in O(len(pattern) + len(text)).
    """
    m = len(pattern)
    if m == 0:
        return 0

    # Build prefix function (pi) for KMP
    pi = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = pi[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        pi[i] = j

    # Scan text, tracking longest prefix match length observed
    q = 0
    best = 0
    for t in text:
        while q > 0 and pattern[q] != t:
            q = pi[q - 1]
        if pattern[q] == t:
            q += 1
        if q > best:
            best = q
        if q == m:
            # Allow overlaps to continue scanning
            q = pi[q - 1]
    return best


@jit(nopython=True)
def suffix_match_score(a: list[int], b: list[int]) -> float:
    assert len(a) == len(b)
    # Suffix of a equals prefix of reversed(a); match anywhere in reversed(b)
    ra = a[::-1]
    rb = b[::-1]
    l1 = longest_prefix_in_text(ra, rb)

    # Suffix of b equals prefix of reversed(b); match anywhere in reversed(a)
    l2 = longest_prefix_in_text(rb, ra)

    return max(l1, l2) / len(a)


@jit(nopython=True)
def lcs_len(a: list[int], b: list[int]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Make 'a' the shorter one
    if m > n:
        a, b = b, a
        m, n = n, m

    prev = [0] * (m + 1)
    curr = [0] * (m + 1)

    for i in range(n):
        for j in range(m):
            if b[i] == a[j]:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(prev[j + 1], curr[j])
        prev, curr = curr, prev
    return prev[m]


@jit(nopython=True)
def rouge_l_score(a: list[int], b: list[int]) -> float:
    """
    Return the ROUGE-L score between a and b.
    """
    lcs = lcs_len(a, b)
    prec = lcs / len(a)
    rec = lcs / len(b)

    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)
