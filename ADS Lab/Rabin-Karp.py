def rabin_karp(text, pattern):
    prime = 101  # A prime number used as the base for hashing
    d = 256  # Number of possible characters (ASCII characters)

    M = len(pattern)
    N = len(text)
    p = 0  # Hash value for the pattern
    t = 0  # Hash value for the current text window

    h = pow(d, M - 1) % prime  # Precompute h, which is (d^(M-1)) % prime

    result = []

    # Calculate the hash values of the pattern and the first window of text
    for i in range(M):
        p = (d * p + ord(pattern[i])) % prime
        t = (d * t + ord(text[i])) % prime

    # Slide the pattern over the text one by one
    for i in range(N - M + 1):
        # Check if the hash values match, and if they do, perform a full string comparison
        if p == t:
            if pattern == text[i : i + M]:
                result.append(i)

        # Calculate the hash value for the next text window
        if i < N - M:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + M])) % prime
            if t < 0:
                t += prime

    return result

# Example usage:
text = "ABABCABAB"
pattern = "ABAB"
result = rabin_karp(text, pattern)

if result:
    print("Pattern found at index(s):", result)
else:
        print("Pattern not found in the text.")
