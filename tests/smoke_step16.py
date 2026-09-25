import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.ingest.transform import SanitizedTag
from src.docs.tags import (
    format_weighted_tags,
    get_top_tag_names,
    split_tags_by_confidence,
    compute_tag_density_score,
    DEFAULT_MIN_RANK,
    DEFAULT_MAX_TAGS,
    HIGH_CONFIDENCE_THRESHOLD,
)

# ── Build a realistic tag set ──────────────────────────────────────────────
def tag(id_, name, rank, spoiler=False):
    return SanitizedTag(id=id_, name=name, rank=rank, is_spoiler=spoiler)

tags = [
    tag(1,  'Military',       88),
    tag(2,  'Gore',           76),
    tag(3,  'Survival',       92),
    tag(4,  'Dark Fantasy',   83),
    tag(5,  'Tragedy',        71),
    tag(6,  'Politics',       65),
    tag(7,  'Plot Twist',     95, spoiler=True),  # must be excluded
    tag(8,  'Isekai',         30),                # below min_rank=40
    tag(9,  'Romance',        55),
    tag(10, 'Action',         88),
    tag(11, 'Mystery',        60),
    tag(12, 'Coming of Age',  48),
]

# ── format_weighted_tags ───────────────────────────────────────────────────
result = format_weighted_tags(tags, min_rank=40, max_tags=5)
print('Weighted tags (top 5, min=40):')
print(' ', result)
assert 'plot twist' not in result.lower(), 'Spoiler leaked!'
assert 'isekai' not in result.lower(), 'Low-rank tag leaked!'
assert result.startswith('survival'), 'Highest rank should be first'

# ── empty / all-filtered case ──────────────────────────────────────────────
empty_result = format_weighted_tags([tag(99, 'Niche', 10)], min_rank=40)
assert empty_result == 'None', f'Expected "None", got {empty_result!r}'
print('Empty fallback:  ', empty_result)

# ── get_top_tag_names ──────────────────────────────────────────────────────
names = get_top_tag_names(tags, n=3)
print('Top 3 names:     ', names)
assert len(names) == 3
assert names[0] == 'survival'

# ── split_tags_by_confidence ───────────────────────────────────────────────
high, supporting = split_tags_by_confidence(tags)
print('High-confidence  :', [t.name for t in high])
print('Supporting        :', [t.name for t in supporting])
assert all(t.rank >= HIGH_CONFIDENCE_THRESHOLD for t in high)
assert all(DEFAULT_MIN_RANK <= t.rank < HIGH_CONFIDENCE_THRESHOLD for t in supporting)

# ── compute_tag_density_score ──────────────────────────────────────────────
density = compute_tag_density_score(tags)
print('Tag density score:', density)
assert 0.0 < density <= 1.0

sparse_density = compute_tag_density_score([])
assert sparse_density == 0.0

print()
print('Step 16 PASS')
