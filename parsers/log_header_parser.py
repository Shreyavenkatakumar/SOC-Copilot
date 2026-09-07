"""
HDFS raw log header extraction.

LogHub HDFS log lines follow the fixed structural format::

    <Date> <Time> <Pid> <Level> <Component>: <Content>

Example::

    081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block
    blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010

Only the free-text ``Content`` portion is fed into Drain3 — mining
templates from the full line (including the ever-changing timestamp and
pid) would prevent the tree from ever converging on stable templates.
"""

import re
from dataclasses import dataclass
from typing import Optional

from utils.exceptions import LogParsingError

# Named-group regex matching the LogHub HDFS header format.
HDFS_HEADER_REGEX = re.compile(
    r"^(?P<Date>\d{6})\s+"
    r"(?P<Time>\d{6})\s+"
    r"(?P<Pid>\d+)\s+"
    r"(?P<Level>[A-Z]+)\s+"
    r"(?P<Component>[^:]+):\s+"
    r"(?P<Content>.*)$"
)

# Block IDs referenced inside the free-text content, e.g. "blk_-1608999687919862906".
BLOCK_ID_REGEX = re.compile(r"blk_-?\d+")


@dataclass
class ParsedHeader:
    """Structured representation of a single raw HDFS log line."""

    date: str
    time: str
    timestamp: str
    pid: str
    level: str
    component: str
    content: str
    block_id: Optional[str]


def extract_header(raw_line: str) -> ParsedHeader:
    """
    Split a raw HDFS log line into its structural header fields and the
    free-text content that should be handed to Drain3.

    Parameters
    ----------
    raw_line : str
        A single raw, CRLF-stripped line from ``HDFS.log``.

    Returns
    -------
    ParsedHeader

    Raises
    ------
    LogParsingError
        If ``raw_line`` does not match the expected HDFS header format.
    """
    match = HDFS_HEADER_REGEX.match(raw_line)
    if not match:
        raise LogParsingError(f"Line does not match HDFS header format: {raw_line!r}")

    fields = match.groupdict()
    content = fields["Content"]
    block_match = BLOCK_ID_REGEX.search(content)

    return ParsedHeader(
        date=fields["Date"],
        time=fields["Time"],
        timestamp=f"{fields['Date']}{fields['Time']}",
        pid=fields["Pid"],
        level=fields["Level"],
        component=fields["Component"],
        content=content,
        block_id=block_match.group(0) if block_match else None,
    )
