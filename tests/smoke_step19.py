import os, tempfile
os.environ['PYTHONIOENCODING'] = 'utf-8'

from pathlib import Path
from src.ingest.transform import SanitizedAnime, SanitizedTag
from src.docs.assembler import build_anime_document, build_document_batch
from src.docs.snapshot import (
    save_processed_corpus, load_processed_corpus, load_corpus_as_dicts,
    load_document_by_id, corpus_exists, get_corpus_stats, load_manifest,
)

def tag(id_, name, rank):
    return SanitizedTag(id=id_, name=name, rank=rank, is_spoiler=False)

def make_anime(id_, title, eng=None):
    return SanitizedAnime(
        id=id_, title_romaji=title, title_english=eng,
        description_clean=f'{title} is a great anime.',
        genres=['Action'], tags=[tag(1,'Survival',90), tag(2,'Drama',75)],
        average_score=85, popularity=500000, season_year=2013,
        format='TV', episodes=24, status='FINISHED', studios=['Studio A'],
    )

catalog = [
    make_anime(16498, 'Shingeki no Kyojin', 'Attack on Titan'),
    make_anime(5114,  'Fullmetal Alchemist: Brotherhood'),
    make_anime(9253,  'Steins;Gate'),
]
root_map = {16498: 16498, 5114: 5114, 9253: 9253}

docs, errors = build_document_batch(catalog, root_map)
assert len(docs) == 3 and len(errors) == 0

# ── Save & load in a temp dir ─────────────────────────────────────────────
with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)

    # corpus_exists → False before save
    assert not corpus_exists(processed_dir=tmp)

    # Save
    out_path = save_processed_corpus(docs, processed_dir=tmp)
    print('Saved corpus to:', out_path.name)

    # corpus_exists → True after save
    assert corpus_exists(processed_dir=tmp)

    # Manifest
    manifest = load_manifest(processed_dir=tmp)
    print('Manifest:', manifest)
    assert manifest['record_count'] == 3
    assert manifest['schema_version'] == '1.0'
    assert 'created_at_utc' in manifest

    # get_corpus_stats
    stats = get_corpus_stats(processed_dir=tmp)
    print('Stats:', stats)
    assert stats['record_count'] == 3
    assert stats['corpus_exists'] is True

    # load_processed_corpus (full round-trip via Pydantic)
    loaded = load_processed_corpus(processed_dir=tmp)
    assert len(loaded) == 3
    assert loaded[0].doc_id == 'anilist_16498'
    assert loaded[0].title == 'Attack on Titan'
    assert loaded[0].average_score == 85
    print('Round-trip load: OK —', loaded[0].doc_id, '|', loaded[0].title)

    # load_corpus_as_dicts (fast dict path)
    raw_dicts = load_corpus_as_dicts(processed_dir=tmp)
    assert len(raw_dicts) == 3
    assert isinstance(raw_dicts[0], dict)
    assert raw_dicts[0]['anime_id'] == 16498

    # load_document_by_id
    single = load_document_by_id(5114, processed_dir=tmp)
    assert single is not None
    assert single.anime_id == 5114
    print('load_document_by_id(5114):', single.metadata.get('title_romaji'))

    # Missing ID returns None
    missing = load_document_by_id(99999, processed_dir=tmp)
    assert missing is None
    print('load_document_by_id(99999):', missing)

print()
print('Step 19 PASS')
