import heapq
from collections import defaultdict

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    frequency = defaultdict(int)
    for char in data:
        frequency[char] += 1

    min_heap = [HuffmanNode(char, freq) for char, freq in frequency.items()]
    heapq.heapify(min_heap)

    while len(min_heap) > 1:
        left = heapq.heappop(min_heap)
        right = heapq.heappop(min_heap)
        merged_node = HuffmanNode(None, left.freq + right.freq)
        merged_node.left = left
        merged_node.right = right
        heapq.heappush(min_heap, merged_node)

    return min_heap[0]

def build_huffman_codes(root, current_code, huffman_codes):
    if root is None:
        return

    if root.char:
        huffman_codes[root.char] = current_code
        return

    build_huffman_codes(root.left, current_code + "0", huffman_codes)
    build_huffman_codes(root.right, current_code + "1", huffman_codes)

def huffman_encoding(data):
    if len(data) == 0:
        return None, None

    root = build_huffman_tree(data)
    huffman_codes = {}
    build_huffman_codes(root, "", huffman_codes)

    encoded_data = "".join(huffman_codes[char] for char in data)
    return encoded_data, root

def huffman_decoding(encoded_data, root):
    if len(encoded_data) == 0:
        return ""

    current_node = root
    decoded_data = ""
    for bit in encoded_data:
        if bit == "0":
            current_node = current_node.left
        else:
            current_node = current_node.right

        if current_node.char:
            decoded_data += current_node.char
            current_node = root

    return decoded_data

# Example usage:
data = "this is an example for huffman encoding"

# Encode the data
encoded_data, huffman_tree = huffman_encoding(data)
print("Encoded data:", encoded_data)

# Decode the encoded data
decoded_data = huffman_decoding(encoded_data, huffman_tree)
print("Decoded data:", decoded_data)
