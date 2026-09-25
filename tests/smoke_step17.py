import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.ingest.transform import SanitizedAnime, SanitizedTag, RelationEdge
from src.docs.assembler import build_document_text, build_anime_document, build_document_batch
from src.docs.schema import MAX_DESCRIPTION_CHARS

# ── Build a realistic SanitizedAnime ─────────────────────────────────────
def tag(id_, name, rank, spoiler=False):
    return SanitizedTag(id=id_, name=name, rank=rank, is_spoiler=spoiler)

aot = SanitizedAnime(
    id=16498,
    id_mal=16498,
    title_romaji='Shingeki no Kyojin',
    title_english='Attack on Titan',
    description_clean=(
        'Several hundred years ago, humans were nearly exterminated by titans. '
        'Titans are typically several stories tall, seem to have no intelligence, '
        'devour human beings and, worst of all, seem to do it for the pleasure '
        'rather than as a food source. A small percentage of humanity survived by '
        'walling themselves in a city protected by extremely high walls.'
    ),
    genres=['Action', 'Drama', 'Fantasy'],
    tags=[
        tag(1, 'Survival',     92),
        tag(2, 'Military',     88),
        tag(3, 'Dark Fantasy', 83),
        tag(4, 'Gore',         76),
        tag(5, 'Plot Twist',   95, spoiler=True),   # must be excluded
        tag(6, 'Isekai',       20),                  # below min_rank
    ],
    average_score=84,
    mean_score=85,
    popularity=600000,
    favourites=120000,
    season_year=2013,
    season='SPRING',
    format='TV',
    episodes=25,
    status='FINISHED',
    studios=['Wit Studio'],
    cover_image_large='https://img.anilist.co/aot.jpg',
    site_url='https://anilist.co/anime/16498',
    relations=[RelationEdge(relation_type='SEQUEL', target_id=20958, target_title_romaji='AoT S2')],
)

# ── Test build_document_text ──────────────────────────────────────────────
text = build_document_text(aot)
print('=== DOCUMENT TEXT ===')
print(text)
print()

# Assertions
assert 'Shingeki no Kyojin' in text
assert 'Attack on Titan' in text
assert '2013' in text
assert 'Spring' in text
assert 'TV' in text
assert '25 episodes' in text
assert 'Action' in text
assert 'survival (92)' in text
assert 'military (88)' in text
assert 'plot twist' not in text.lower(), 'Spoiler leaked!'
assert 'isekai' not in text.lower(), 'Low-rank tag leaked!'
assert '84/100' in text
assert '600000' in text
assert len(text) <= MAX_DESCRIPTION_CHARS * 2, 'Document unexpectedly large'

# ── Test same-title deduplication ─────────────────────────────────────────
same_title = SanitizedAnime(
    id=99, title_romaji='Naruto', title_english='Naruto',
    description_clean='A ninja story.', genres=['Action'],
)
t2 = build_document_text(same_title)
assert 'Naruto (Naruto)' not in t2, 'Duplicate title should not appear'
print('Dedup check passed:', t2.split('\n')[0])

# ── Test movie (no episodes) ───────────────────────────────────────────────
movie = SanitizedAnime(
    id=100, title_romaji='Spirited Away', title_english='Spirited Away',
    description_clean='A girl enters a spirit world.',
    format='MOVIE', episodes=None, genres=['Adventure'],
)
t3 = build_document_text(movie)
assert 'episodes' not in t3
print('Movie check passed:', t3.split('\n')[0])

# ── Test full AnimeDocument build ──────────────────────────────────────────
doc = build_anime_document(aot, franchise_root_id=16498)
print()
print('=== AnimeDocument ===')
print('doc_id          :', doc.doc_id)
print('franchise_root  :', doc.franchise_root_id)
print('title           :', doc.title)
print('average_score   :', doc.average_score)
print('season_year     :', doc.season_year)
print('metadata keys   :', sorted(doc.metadata.keys()))

assert doc.doc_id == 'anilist_16498'
assert doc.metadata['franchise_root_id'] == 16498
assert doc.metadata['studio'] == 'Wit Studio'
assert 'plot twist' not in doc.metadata.get('top_tags', '')
assert doc.metadata['tag_density'] > 0

# ── Test batch builder ─────────────────────────────────────────────────────
catalog = [aot, same_title, movie]
root_map = {16498: 16498, 99: 99, 100: 100}
docs, errors = build_document_batch(catalog, root_map)
assert len(docs) == 3
assert len(errors) == 0
print()
print('Batch built:', len(docs), 'documents,', len(errors), 'errors')
print()
print('Step 17 PASS')
