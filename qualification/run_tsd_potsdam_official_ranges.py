from __future__ import annotations

import re
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from typing import Final

from qualification import download_tsd_potsdam_official_ranges as impl

MAX_RANGE_ATTEMPTS: Final = 5
RANGE_OPEN_TIMEOUT_SECONDS: Final = 90
BACKOFF_SECONDS: Final[tuple[float, ...]] = (1.0, 2.0, 4.0, 8.0)
RETRYABLE_HTTP_STATUS: Final[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504})


class ResilientHttpRangeSource(impl.HttpRangeSource):
    """The scientific range source with bounded retries for transient transport failure only."""

    retry_attempts_total = 0

    def _open_range(self, start: int, end: int):
        if start < 0 or end < start:
            raise ValueError(f"invalid HTTP range {start}-{end}")

        last_error: BaseException | None = None
        for attempt in range(1, MAX_RANGE_ATTEMPTS + 1):
            request = urllib.request.Request(
                impl.DOWNLOAD_URL,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                    "Referer": impl.SHARE_URL,
                    "User-Agent": impl.USER_AGENT,
                },
            )
            try:
                response = self.opener.open(request, timeout=RANGE_OPEN_TIMEOUT_SECONDS)
                status = getattr(response, "status", response.getcode())
                content_range = response.headers.get("Content-Range", "")
                if status != 206:
                    response.close()
                    raise RuntimeError(
                        "official Potsdam server did not honor the requested byte range; refusing any "
                        f"fallback that could download the complete archive (status={status})"
                    )
                match = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+|\*)",
                    content_range.strip(),
                    flags=re.IGNORECASE,
                )
                if match is None:
                    response.close()
                    raise RuntimeError(f"unexpected Content-Range header: {content_range!r}")
                actual_start = int(match.group(1))
                actual_end = int(match.group(2))
                if (actual_start, actual_end) != (start, end):
                    response.close()
                    raise RuntimeError(
                        "official server returned a different range than requested: "
                        f"requested={start}-{end}, returned={actual_start}-{actual_end}"
                    )
                if match.group(3) != "*":
                    total_size = int(match.group(3))
                    if self.total_size is None:
                        self.total_size = total_size
                    elif self.total_size != total_size:
                        response.close()
                        raise RuntimeError("remote Potsdam.zip size changed during acquisition")
                self.requests += 1
                return response
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRYABLE_HTTP_STATUS:
                    raise
                last_error = exc
            except (urllib.error.URLError, TimeoutError, socket.timeout, ssl.SSLError) as exc:
                last_error = exc

            if attempt == MAX_RANGE_ATTEMPTS:
                break
            type(self).retry_attempts_total += 1
            delay = BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)]
            print(
                "TRANSIENT_RANGE_RETRY "
                f"attempt={attempt + 1}/{MAX_RANGE_ATTEMPTS} "
                f"range={start}-{end} wait_s={delay:g} "
                f"reason={type(last_error).__name__}: {last_error}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

        assert last_error is not None
        raise last_error


def main() -> int:
    impl.HttpRangeSource = ResilientHttpRangeSource
    result = impl.main()
    print(f"transient_range_retries={ResilientHttpRangeSource.retry_attempts_total}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
