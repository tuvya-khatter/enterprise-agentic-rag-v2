"""Document chunking: recursive character splitter with token-aware overlap."""
import tiktoken


class Chunker:
    """Token-aware recursive text splitter."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding(encoding_name)

    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def _split_on_separator(self, text: str, separator: str) -> list[str]:
        if separator == "":
            return list(text)
        return [part for part in text.split(separator) if part]

    def split(self, text: str) -> list[str]:
        """Split text into token-bounded chunks with overlap, preferring larger natural boundaries."""
        separators = ["\n\n", "\n", ". ", " ", ""]
        chunks: list[str] = []
        self._recursive_split(text, separators, chunks)
        return self._merge_with_overlap(chunks)

    def _recursive_split(self, text: str, separators: list[str], out: list[str]) -> None:
        if self._count_tokens(text) <= self.chunk_size:
            out.append(text)
            return

        if not separators:
            out.append(text)
            return

        sep = separators[0]
        parts = self._split_on_separator(text, sep)

        if len(parts) == 1:
            self._recursive_split(text, separators[1:], out)
            return

        for part in parts:
            if self._count_tokens(part) <= self.chunk_size:
                out.append(part)
            else:
                self._recursive_split(part, separators[1:], out)

    def _merge_with_overlap(self, parts: list[str]) -> list[str]:
        """Merge small parts into near-target-sized chunks with overlap."""
        merged: list[str] = []
        buffer: list[str] = []
        buffer_tokens = 0

        for part in parts:
            t = self._count_tokens(part)
            if buffer_tokens + t <= self.chunk_size:
                buffer.append(part)
                buffer_tokens += t
            else:
                if buffer:
                    merged.append(" ".join(buffer))
                buffer = [part]
                buffer_tokens = t

        if buffer:
            merged.append(" ".join(buffer))

        # Add overlap
        if self.chunk_overlap == 0 or len(merged) < 2:
            return merged
        with_overlap: list[str] = [merged[0]]
        for i in range(1, len(merged)):
            prev_tokens = self.encoder.encode(merged[i - 1])
            overlap_tokens = prev_tokens[-self.chunk_overlap :] if self.chunk_overlap else []
            overlap_text = self.encoder.decode(overlap_tokens) if overlap_tokens else ""
            with_overlap.append((overlap_text + " " + merged[i]).strip())
        return with_overlap
