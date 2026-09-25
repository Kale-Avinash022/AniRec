import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.ingest.transform import SanitizedAnime, SanitizedTag
from src.docs.metadata import (
    build_vector_metadata, validate_metadata,
    build_where_clause, parse_genres_from_metadata,
    parse_top_tags_from_metadata, METADATA_FIELD_TYPES,
    RANGE_FILTER_FIELDS, EQUALITY_FILTER_FIELDS,
)

def tag(id_, name, rank, spoiler=False):
    return SanitizedTag(id=id_, name=name, rank=rank, is_spoiler=spoiler)

aot = SanitizedAnime(
    id=16498, id_mal=16498,
    title_romaji='Shingeki no Kyojin', title_english='Attack on Titan',
    description_clean='Humans vs titans.',
    genres=['Action', 'Drama', 'Fantasy'],
    tags=[tag(1,'Survival',92), tag(2,'Military',88), tag(3,'Gore',76), tag(4,'Nope',10)],
    average_score=84, mean_score=85, popularity=600000, favourites=120000,
    season_year=2013, season='SPRING', format='TV', episodes=25,
    status='FINISHED', studios=['Wit Studio'],
    cover_image_large='https://img.anilist.co/aot.jpg',
    site_url='https://anilist.co/anime/16498',
)

# ── build_vector_metadata ─────────────────────────────────────────────────
meta = build_vector_metadata(aot, franchise_root_id=16498)
print('=== Metadata dict ===')
for k, v in sorted(meta.items()):
    print(f'  {k:<22} = {v!r}  ({type(v).__name__})')

# Type checks
assert meta['anime_id'] == 16498
assert meta['season_year'] == 2013
assert meta['average_score'] == 84
assert meta['genres'] == 'Action,Drama,Fantasy'
assert 'survival' in meta['top_tags']
assert 'nope' not in meta['top_tags'], 'Low-rank tag should be filtered'
assert isinstance(meta['tag_density'], float)
assert meta['franchise_root_id'] == 16498
assert meta['studio'] == 'Wit Studio'
# No None values
assert all(v is not None for v in meta.values()), 'None found in metadata!'

# ── validate_metadata ─────────────────────────────────────────────────────
violations = validate_metadata(meta)
assert violations == [], f'Schema violations: {violations}'
print('\nValidation: CLEAN (0 violations)')

# Bad metadata — wrong type
bad = dict(meta)
bad['average_score'] = '84'   # str instead of int
bad['season_year'] = None     # None not allowed
violations2 = validate_metadata(bad)
print(f'Bad metadata violations: {len(violations2)} (expected >= 2)')
assert len(violations2) >= 2

# ── build_where_clause ────────────────────────────────────────────────────
print('\n=== WHERE clauses ===')

# No filters → None
w0 = build_where_clause()
assert w0 is None
print('No filters:', w0)

# Single filter → unwrapped (no $and)
w1 = build_where_clause(format_filter='TV')
print('Format only:', w1)
assert w1 == {'format': {'$eq': 'TV'}}

# Multiple filters → $and wrapped
w2 = build_where_clause(format_filter='TV', min_score=80, min_year=2015)
print('TV + score + year:', w2)
assert '$and' in w2
assert {'average_score': {'$gte': 80}} in w2['$and']
assert {'season_year': {'$gte': 2015}} in w2['$and']

# Franchise exclusion (Step 30 dedup)
w3 = build_where_clause(exclude_franchise_root_ids=[16498, 20958])
print('Exclude franchise roots:', w3)
assert {'franchise_root_id': {'$ne': 16498}} in w3['$and']

# ── parse helpers ─────────────────────────────────────────────────────────
genres = parse_genres_from_metadata(meta)
assert genres == ['Action', 'Drama', 'Fantasy'], f'Got: {genres}'

tags = parse_top_tags_from_metadata(meta)
assert 'survival' in tags

print('\nField registry:', len(METADATA_FIELD_TYPES), 'fields defined')
print('Range fields  :', len(RANGE_FILTER_FIELDS))
print('Equality fields:', len(EQUALITY_FILTER_FIELDS))
print()
print('Step 18 PASS')
