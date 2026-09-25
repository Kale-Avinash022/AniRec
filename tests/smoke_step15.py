import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.docs.schema import (
    AnimeDocument, CHUNKING_STRATEGY, MAX_TAG_SLOTS,
    MAX_DESCRIPTION_CHARS, DOCUMENT_VERSION,
    FRANCHISE_RELATION_LABEL, CHUNKING_STRATEGY_COMPARISON
)

# Verify constants
assert CHUNKING_STRATEGY == 'single_doc_per_show'
assert MAX_TAG_SLOTS == 15
assert MAX_DESCRIPTION_CHARS == 1200
assert DOCUMENT_VERSION == '1.0'
assert FRANCHISE_RELATION_LABEL == 'franchise_root_id'

# Verify AnimeDocument schema
doc = AnimeDocument(
    doc_id=AnimeDocument.make_doc_id(16498),
    anime_id=16498,
    franchise_root_id=16498,
    document_text='Attack on Titan | 2013 | Action, Drama | Dark Fantasy (95), Military (88) | A world where...',
    metadata={
        'title_romaji': 'Shingeki no Kyojin',
        'title_english': 'Attack on Titan',
        'average_score': 90,
        'season_year': 2013,
        'format': 'TV',
        'genres': 'Action,Drama',
        'franchise_root_id': 16498,
    }
)

print('doc_id         :', doc.doc_id)
print('title          :', doc.title)
print('average_score  :', doc.average_score)
print('season_year    :', doc.season_year)
print('schema_version :', doc.schema_version)
print('doc_text[:60]  :', doc.document_text[:60])
print()
print('Chunking strategy comparison:')
for row in CHUNKING_STRATEGY_COMPARISON:
    verdict = row['verdict'][:8]
    approach = row['approach']
    print(f'  [{verdict}] {approach}')
print()
print('Step 15 PASS')
